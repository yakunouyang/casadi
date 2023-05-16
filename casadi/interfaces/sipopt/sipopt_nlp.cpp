/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "sipopt_nlp.hpp"
#include "sipopt_interface.hpp"
#include <casadi/core/timing.hpp>
#include <casadi/core/convexify.hpp>
#include <IpDenseVector.hpp>

namespace casadi {

  SIpoptUserClass::SIpoptUserClass(const SIpoptInterface& solver, SIpoptMemory* mem)
      : solver_(solver), mem_(mem) {
    n_ = solver_.nx_ + solver_.np_;
    m_ = solver_.ng_ + solver_.np_;

#ifdef WITH_IPOPT_CALLBACK
    x_ = new double[n_];
    g_ = new double[m_];
    z_L_ = new double[n_];
    z_U_ = new double[n_];
    lambda_ = new double[m_];
#endif // WITH_IPOPT_CALLBACK

  }

  SIpoptUserClass::~SIpoptUserClass() {
#ifdef WITH_IPOPT_CALLBACK
    delete [] x_;
    delete [] g_;
    delete [] z_U_;
    delete [] z_L_;
    delete [] lambda_;
#endif // WITH_IPOPT_CALLBACK
  }

  // returns the size of the problem
  bool SIpoptUserClass::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                    Index& nnz_h_lag, IndexStyleEnum& index_style) {
    solver_.get_nlp_info(mem_, n, m, nnz_jac_g, nnz_h_lag);

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;

    return true;
  }

  // returns the variable bounds
  bool SIpoptUserClass::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                       Index m, Number* g_l, Number* g_u) {
    casadi_assert_dev(n==solver_.nx_+solver_.np_);
    casadi_assert_dev(m==solver_.ng_+solver_.np_);
    return solver_.get_bounds_info(mem_, x_l, x_u, g_l, g_u);
  }

  // returns the initial point for the problem
  bool SIpoptUserClass::get_starting_point(Index n, bool init_x, Number* x,
                                          bool init_z, Number* z_L, Number* z_U,
                                          Index m, bool init_lambda,
                                          Number* lambda) {
    casadi_assert_dev(n==solver_.nx_+solver_.np_);
    casadi_assert_dev(m==solver_.ng_+solver_.np_);
    return solver_.get_starting_point(mem_, init_x, x, init_z, z_L, z_U, init_lambda, lambda);
  }

  // returns the value of the objective function
  bool SIpoptUserClass::eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
    mem_->arg[0] = x;
    mem_->arg[1] = x + solver_.nx_;
    mem_->res[0] = &obj_value;
    try {
      return solver_.calc_function(mem_, "nlp_f")==0;
    } catch(KeyboardInterruptException& ex) {
      casadi_warning("KeyboardInterruptException");
      throw KeyboardInterruptException();
    } catch (std::exception& ex) {
      if (solver_.show_eval_warnings_) {
        casadi_warning("SIpoptUserClass::eval_f failed:" + std::string(ex.what()));
      }
      return false;
    }
  }

  // return the gradient of the objective function grad_ {x} f(x)
  bool SIpoptUserClass::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
    mem_->arg[0] = x;
    mem_->arg[1] = x + solver_.nx_;
    mem_->res[0] = nullptr;
    mem_->res[1] = grad_f;
    mem_->res[2] = grad_f + solver_.nx_;
    try {
      return solver_.calc_function(mem_, "nlp_grad_f")==0;
    } catch(KeyboardInterruptException& ex) {
      casadi_warning("KeyboardInterruptException");
      throw KeyboardInterruptException();
    }  catch (std::exception& ex) {
      if (solver_.show_eval_warnings_) {
        casadi_warning("SIpoptUserClass::eval_grad_f failed:" + std::string(ex.what()));
      }
      return false;
    }
  }

  // return the value of the constraints: g(x)
  bool SIpoptUserClass::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
    mem_->arg[0] = x;
    mem_->arg[1] = x + solver_.nx_;
    mem_->res[0] = g;
    // Direct copy parameter values to tail constraint values
    casadi_copy(x + solver_.nx_, solver_.np_, g + solver_.ng_);
    try {
      return solver_.calc_function(mem_, "nlp_g")==0;
    } catch(KeyboardInterruptException& ex) {
      casadi_warning("KeyboardInterruptException");
      throw KeyboardInterruptException();
    } catch (std::exception& ex) {
      if (solver_.show_eval_warnings_) {
        casadi_warning("SIpoptUserClass::eval_g failed:" + std::string(ex.what()));
      }
      return false;
    }
  }

  // return the structure or values of the jacobian
  bool SIpoptUserClass::eval_jac_g(Index n, const Number* x, bool new_x,
                                  Index m, Index nele_jac, Index* iRow, Index *jCol,
                                  Number* values) {
    auto jacg_nnz = solver_.jacg_sp_.nnz();
    auto jacgp_nnz = solver_.jacgp_sp_.nnz();
    casadi_assert_dev(jacg_nnz + jacgp_nnz + solver_.np_ == nele_jac); // consistency check

    if (values) {
      // Evaluate numerically
      mem_->arg[0] = x;
      mem_->arg[1] = x + solver_.nx_;
      mem_->res[0] = nullptr;
      mem_->res[1] = values;
      mem_->res[2] = values + jacg_nnz;
      // Diag values
      std::fill_n(values + jacg_nnz + jacgp_nnz, solver_.np_, 1.0);
      try {
        return solver_.calc_function(mem_, "nlp_jac_g")==0;
      } catch(KeyboardInterruptException& ex) {
        casadi_warning("KeyboardInterruptException");
        throw KeyboardInterruptException();
      } catch (std::exception& ex) {
        if (solver_.show_eval_warnings_) {
          casadi_warning("SIpoptUserClass::eval_jac_g failed:" + std::string(ex.what()));
        }
        return false;
      }
    } else {
      // Get the sparsity pattern as triplets
      std::vector<casadi_int> jacg_rows, jacg_cols, jacgp_rows, jacgp_cols;
      solver_.jacg_sp_.get_triplet(jacg_rows, jacg_cols);
      solver_.jacgp_sp_.get_triplet(jacgp_rows, jacgp_cols);

      // Parameters are indexed after variables
      for(auto &c: jacgp_cols) {
        c += solver_.nx_;
      }

      std::copy(jacg_rows.begin(), jacg_rows.end(), iRow);
      std::copy(jacg_cols.begin(), jacg_cols.end(), jCol);
      std::copy(jacgp_rows.begin(), jacgp_rows.end(), iRow + jacg_nnz);
      std::copy(jacgp_cols.begin(), jacgp_cols.end(), jCol + jacg_nnz);

      // Diag values
      for(casadi_int i = 0; i < solver_.np_; i++) {
        iRow[jacg_nnz + jacgp_nnz + i] = solver_.ng_ + i;
        jCol[jacg_nnz + jacgp_nnz + i] = solver_.nx_ + i;
      }
      return true;
    }
  }


  bool SIpoptUserClass::eval_h(Index n, const Number* x, bool new_x,
                              Number obj_factor, Index m, const Number* lambda,
                              bool new_lambda, Index nele_hess, Index* iRow,
                              Index* jCol, Number* values) {
    auto hesslag_nnz = solver_.hesslag_sp_.nnz();
    if (values) {
      // Evaluate numerically
      mem_->arg[0] = x;
      mem_->arg[1] = x + solver_.nx_;
      mem_->arg[2] = &obj_factor;
      mem_->arg[3] = lambda;
      mem_->res[0] = values;
      mem_->res[1] = values + hesslag_nnz;
      try {
        if (solver_.calc_function(mem_, "nlp_hess_l")) return false;
      } catch(KeyboardInterruptException& ex) {
        casadi_warning("KeyboardInterruptException");
        throw KeyboardInterruptException();
      } catch (std::exception& ex) {
        if (solver_.show_eval_warnings_) {
          casadi_warning("SIpoptUserClass::eval_h failed:" + std::string(ex.what()));
        }
        return false;
      }
      if (solver_.convexify_) {
        ScopedTiming tic(mem_->fstats.at("convexify"));
        if (convexify_eval(&solver_.convexify_data_.config, values, values, mem_->iw, mem_->w)) {
          return false;
        }
      }
      return true;
    } else {
      auto hesslagp_nnz = solver_.hesslagp_sp_.nnz();
      casadi_assert(hesslag_nnz + hesslagp_nnz == nele_hess, "nnz count error"); // consistency check

      // Get the sparsity pattern as triplets
      std::vector<casadi_int> hesslag_rows, hesslag_cols, hesslagp_rows, hesslagp_cols;
      solver_.hesslag_sp_.get_triplet(hesslag_rows, hesslag_cols);
      solver_.hesslagp_sp_.get_triplet(hesslagp_rows, hesslagp_cols);

      // Parameters are indexed after variables
      for(auto &c: hesslagp_cols) {
        c += solver_.nx_;
      }

      std::copy(hesslag_rows.begin(), hesslag_rows.end(), iRow);
      std::copy(hesslag_cols.begin(), hesslag_cols.end(), jCol);
      std::copy(hesslagp_rows.begin(), hesslagp_rows.end(), iRow + hesslag_nnz);
      std::copy(hesslagp_cols.begin(), hesslagp_cols.end(), jCol + hesslag_nnz);
      return true;
    }
  }


  void SIpoptUserClass::finalize_solution(SolverReturn status, Index n, const Number* x,
                                         const Number* z_L, const Number* z_U,
                                         Index m, const Number* g, const Number* lambda,
                                         Number obj_value,
                                         const IpoptData* ip_data,
                                         IpoptCalculatedQuantities* ip_cq) {
    std::vector<double> perturbed_x;

    if(solver_.run_sens_) {
      // Get access to the metadata, where the solutions are stored. The metadata is part of the DenseVectorSpace.
      SmartPtr<const DenseVectorSpace> x_owner_space = dynamic_cast<const DenseVectorSpace*>(GetRawPtr(
          ip_data->curr()->x()->OwnerSpace()));
      casadi_assert(IsValid(x_owner_space), "Error IsValid(x_owner_space) failed");

      perturbed_x = x_owner_space->GetNumericMetaData("sens_sol_state_1");
    }

    solver_.finalize_solution(mem_, x, z_L, z_U, g, lambda, obj_value,
                              ip_data? ip_data->iter_count(): 0, perturbed_x);
  }


  bool SIpoptUserClass::intermediate_callback(AlgorithmMode mode, Index iter, Number obj_value,
                                             Number inf_pr, Number inf_du,
                                             Number mu, Number d_norm,
                                             Number regularization_size,
                                             Number alpha_du, Number alpha_pr,
                                             Index ls_trials,
                                             const IpoptData* ip_data,
                                             IpoptCalculatedQuantities* ip_cq) {

    // Only do the callback every few iterations
    if (iter % solver_.callback_step_!=0) return true;

    /// Code copied from TNLPAdapter::FinalizeSolution
    /// See also: http://list.coin-or.org/pipermail/ipopt/2010-July/002078.html
    // http://list.coin-or.org/pipermail/ipopt/2010-April/001965.html

    bool full_callback = false;

#ifdef WITH_IPOPT_CALLBACK
    OrigIpoptNLP* orignlp = dynamic_cast<OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
    if (!orignlp) return true;
    TNLPAdapter* tnlp_adapter = dynamic_cast<TNLPAdapter*>(GetRawPtr(orignlp->nlp()));
    if (!tnlp_adapter) return true;

    const Vector& x = *ip_data->curr()->x();
    const Vector& z_L = *ip_data->curr()->z_L();
    const Vector& z_U = *ip_data->curr()->z_U();
    const Vector& c = *ip_cq->curr_c();
    const Vector& d = *ip_cq->curr_d();
    const Vector& y_c = *ip_data->curr()->y_c();
    const Vector& y_d = *ip_data->curr()->y_d();

    std::fill_n(x_, n_, 0);
    std::fill_n(g_, m_, 0);
    std::fill_n(z_L_, n_, 0);
    std::fill_n(z_U_, n_, 0);
    std::fill_n(lambda_, m_, 0);

    tnlp_adapter->ResortX(x, x_);             // no further steps needed
    tnlp_adapter->ResortG(y_c, y_d, lambda_); // no further steps needed
    tnlp_adapter->ResortG(c, d, g_);
    // Copied from Ipopt source: To Ipopt, the equality constraints are presented with right
    // hand side zero, so we correct for the original right hand side.
    const Index* c_pos = tnlp_adapter->P_c_g_->ExpandedPosIndices();
    Index n_c_no_fixed = tnlp_adapter->P_c_g_->NCols();
    for (Index i=0; i<n_c_no_fixed; i++) {
      g_[c_pos[i]] += tnlp_adapter->c_rhs_[i];
    }

    tnlp_adapter->ResortBounds(z_L, z_L_, z_U, z_U_);
    // Copied from Ipopt source: Hopefully the following is correct to recover the bound
    // multipliers for fixed variables (sign ok?)
    if (tnlp_adapter->fixed_variable_treatment_==TNLPAdapter::MAKE_CONSTRAINT &&
        tnlp_adapter->n_x_fixed_>0) {
      const DenseVector* dy_c = static_cast<const DenseVector*>(&y_c);
      Index n_c_no_fixed = y_c.Dim() - tnlp_adapter->n_x_fixed_;
      if (!dy_c->IsHomogeneous()) {
        const Number* values = dy_c->Values();
        for (Index i=0; i<tnlp_adapter->n_x_fixed_; i++) {
          z_L_[tnlp_adapter->x_fixed_map_[i]] = Max(0., -values[n_c_no_fixed+i]);
          z_U_[tnlp_adapter->x_fixed_map_[i]] = Max(0., values[n_c_no_fixed+i]);
        }
      } else {
        double value = dy_c->Scalar();
        for (Index i=0; i<tnlp_adapter->n_x_fixed_; i++) {
          z_L_[tnlp_adapter->x_fixed_map_[i]] = Max(0., -value);
          z_U_[tnlp_adapter->x_fixed_map_[i]] = Max(0.,  value);
        }
      }
    }
    full_callback = true;
#endif // WITH_IPOPT_CALLBACK

    return solver_.intermediate_callback(mem_, x_, z_L_, z_U_, g_, lambda_, obj_value, iter,
                                         inf_pr, inf_du, mu, d_norm, regularization_size,
                                         alpha_du, alpha_pr, ls_trials, full_callback);
  }

  Index SIpoptUserClass::get_number_of_nonlinear_variables() {
    return solver_.get_number_of_nonlinear_variables();
  }

  bool SIpoptUserClass::get_list_of_nonlinear_variables(Index num_nonlin_vars,
                                                       Index* pos_nonlin_vars) {
    return solver_.get_list_of_nonlinear_variables(num_nonlin_vars, pos_nonlin_vars);
  }

  bool SIpoptUserClass::get_var_con_metadata(Index n, StringMetaDataMapType& var_string_md,
                                            IntegerMetaDataMapType& var_integer_md,
                                            NumericMetaDataMapType& var_numeric_md,
                                            Index m, StringMetaDataMapType& con_string_md,
                                            IntegerMetaDataMapType& con_integer_md,
                                            NumericMetaDataMapType& con_numeric_md) {

    return solver_.get_var_con_metadata(mem_, var_string_md, var_integer_md, var_numeric_md,
                                        con_string_md, con_integer_md, con_numeric_md);
  }

  void SIpoptUserClass::finalize_metadata(Index n, const StringMetaDataMapType& var_string_md,
                                         const IntegerMetaDataMapType& var_integer_md,
                                         const NumericMetaDataMapType& var_numeric_md,
                                         Index m, const StringMetaDataMapType& con_string_md,
                                         const IntegerMetaDataMapType& con_integer_md,
                                         const NumericMetaDataMapType& con_numeric_md) {
    casadi_assert_dev(n==solver_.nx_+solver_.np_);
    casadi_assert_dev(m==solver_.ng_+solver_.np_);
    mem_->var_string_md = var_string_md;
    mem_->var_integer_md = var_integer_md;
    mem_->var_numeric_md = var_numeric_md;
    mem_->con_string_md = con_string_md;
    mem_->con_integer_md = con_integer_md;
    mem_->con_numeric_md = con_numeric_md;
  }

} // namespace casadi
