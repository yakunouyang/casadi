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



#include "sipopt_interface.hpp"
#include "sipopt_nlp.hpp"
#include "casadi/core/casadi_misc.hpp"
#include "casadi/core/global_options.hpp"
#include "casadi/core/casadi_interrupt.hpp"
#include "casadi/core/convexify.hpp"

#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <IpIpoptApplication.hpp>
#include <SensApplication.hpp>
#include <SensRegOp.hpp>

namespace casadi {
  extern "C"
  int CASADI_NLPSOL_SIPOPT_EXPORT
  casadi_register_nlpsol_sipopt(Nlpsol::Plugin* plugin) {
  plugin->creator = SIpoptInterface::creator;
  plugin->name = "sipopt";
  plugin->doc = SIpoptInterface::meta_doc.c_str();
  plugin->version = 1;
  plugin->options = &SIpoptInterface::options_;
  plugin->deserialize = &SIpoptInterface::deserialize;
  return 0;
}

extern "C"
void CASADI_NLPSOL_SIPOPT_EXPORT casadi_load_nlpsol_sipopt() {
  Nlpsol::registerPlugin(casadi_register_nlpsol_sipopt);
}

SIpoptInterface::SIpoptInterface(const std::string& name, const Function& nlp)
    : Nlpsol(name, nlp) {
}

SIpoptInterface::~SIpoptInterface() {
  clear_mem();
}

const Options SIpoptInterface::options_
    = {{&Nlpsol::options_},
       {{"pass_nonlinear_variables",
            {OT_BOOL,
                "Pass list of variables entering nonlinearly to IPOPT"}},
           {"sipopt",
               {OT_DICT,
                   "Options to be passed to sIPOPT"}},
           {"var_string_md",
               {OT_DICT,
                   "String metadata (a dictionary with lists of strings) "
                   "about variables to be passed to IPOPT"}},
           {"var_integer_md",
               {OT_DICT,
                   "Integer metadata (a dictionary with lists of integers) "
                   "about variables to be passed to IPOPT"}},
           {"var_numeric_md",
               {OT_DICT,
                   "Numeric metadata (a dictionary with lists of reals) about "
                   "variables to be passed to IPOPT"}},
           {"con_string_md",
               {OT_DICT,
                   "String metadata (a dictionary with lists of strings) about "
                   "constraints to be passed to IPOPT"}},
           {"con_integer_md",
               {OT_DICT,
                   "Integer metadata (a dictionary with lists of integers) "
                   "about constraints to be passed to IPOPT"}},
           {"con_numeric_md",
               {OT_DICT,
                   "Numeric metadata (a dictionary with lists of reals) about "
                   "constraints to be passed to IPOPT"}},
           {"hess_lag",
               {OT_FUNCTION,
                   "Function for calculating the Hessian of the Lagrangian (autogenerated by default)"}},
           {"jac_g",
               {OT_FUNCTION,
                   "Function for calculating the Jacobian of the constraints "
                   "(autogenerated by default)"}},
           {"grad_f",
               {OT_FUNCTION,
                   "Function for calculating the gradient of the objective "
                   "(column, autogenerated by default)"}},
           {"convexify_strategy",
               {OT_STRING,
                   "NONE|regularize|eigen-reflect|eigen-clip. "
                   "Strategy to convexify the Lagrange Hessian before passing it to the solver."}},
           {"convexify_margin",
               {OT_DOUBLE,
                   "When using a convexification strategy, make sure that "
                   "the smallest eigenvalue is at least this (default: 1e-7)."}},
           {"max_iter_eig",
               {OT_DOUBLE,
                   "Maximum number of iterations to compute an eigenvalue decomposition (default: 50)."}},
           {"clip_inactive_lam",
               {OT_BOOL,
                   "Explicitly set Lagrange multipliers to 0 when bound is deemed inactive "
                   "(default: false)."}},
           {"inactive_lam_strategy",
               {OT_STRING,
                   "Strategy to detect if a bound is inactive. "
                   "RELTOL: use solver-defined constraint tolerance * inactive_lam_value|"
                   "abstol: use inactive_lam_value"}},
           {"inactive_lam_value",
               {OT_DOUBLE,
                   "Value used in inactive_lam_strategy (default: 10)."}},
           {"perturbed_p",
               {OT_DOUBLEVECTOR,
                   "Value  (default: original parameters)."}},
       }
    };

void SIpoptInterface::init(const Dict& opts) {
  // Call the init method of the base class
  Nlpsol::init(opts);

  // Default options
  pass_nonlinear_variables_ = false;

  std::string convexify_strategy = "none";
  double convexify_margin = 1e-7;
  casadi_int max_iter_eig = 200;

  clip_inactive_lam_ = false;
  inactive_lam_strategy_ = "reltol";
  inactive_lam_value_ = 10;

  // Read user options
  for (auto&& op : opts) {
    if (op.first=="sipopt") {
      opts_ = op.second;
    } else if (op.first=="pass_nonlinear_variables") {
      pass_nonlinear_variables_ = op.second;
    } else if (op.first=="var_string_md") {
      var_string_md_ = op.second;
    } else if (op.first=="var_integer_md") {
      var_integer_md_ = op.second;
    } else if (op.first=="var_numeric_md") {
      var_numeric_md_ = op.second;
    } else if (op.first=="con_string_md") {
      con_string_md_ = op.second;
    } else if (op.first=="con_integer_md") {
      con_integer_md_ = op.second;
    } else if (op.first=="con_numeric_md") {
      con_numeric_md_ = op.second;
    } else if (op.first=="hess_lag") {
      Function f = op.second;
      casadi_assert_dev(f.n_in()==4);
      casadi_assert_dev(f.n_out()==1);
      set_function(f, "nlp_hess_l");
    } else if (op.first=="jac_g") {
      Function f = op.second;
      casadi_assert_dev(f.n_in()==2);
      casadi_assert_dev(f.n_out()==2);
      set_function(f, "nlp_jac_g");
    } else if (op.first=="grad_f") {
      Function f = op.second;
      casadi_assert_dev(f.n_in()==2);
      casadi_assert_dev(f.n_out()==2);
      set_function(f, "nlp_grad_f");
    } else if (op.first=="convexify_strategy") {
      convexify_strategy = op.second.to_string();
    } else if (op.first=="convexify_margin") {
      convexify_margin = op.second;
    } else if (op.first=="max_iter_eig") {
      max_iter_eig = op.second;
    } else if (op.first=="clip_inactive_lam") {
      clip_inactive_lam_ = op.second;
    } else if (op.first=="inactive_lam_strategy") {
      inactive_lam_strategy_ = op.second.to_string();
    } else if (op.first=="inactive_lam_value") {
      inactive_lam_value_ = op.second;
    } else if (op.first=="perturbed_p") {
      perturbed_p_ = op.second;
    }
  }

  // Do we need second order derivatives?
  exact_hessian_ = true;
  auto hessian_approximation = opts_.find("hessian_approximation");
  if (hessian_approximation!=opts_.end()) {
    exact_hessian_ = hessian_approximation->second == "exact";
  }

  run_sens_ = false;
  auto run_sens = opts_.find("run_sens");
  if(run_sens != opts_.end()) {
    run_sens_ = run_sens->second == "yes";
  }

  // Setup NLP functions
  create_function("nlp_f", {"x", "p"}, {"f"});
  create_function("nlp_g", {"x", "p"}, {"g"});
  if (!has_function("nlp_grad_f")) {
    create_function("nlp_grad_f", {"x", "p"}, {"f", "grad:f:x", "grad:f:p"});
  }
  if (!has_function("nlp_jac_g")) {
    create_function("nlp_jac_g", {"x", "p"}, {"g", "jac:g:x", "jac:g:p"});
  }
  jacg_sp_ = get_function("nlp_jac_g").sparsity_out(1);
  jacgp_sp_ = get_function("nlp_jac_g").sparsity_out(2);

  convexify_ = false;

  // Allocate temporary work vectors
  if (exact_hessian_) {
    if (!has_function("nlp_hess_l")) {
      create_function("nlp_hess_l", {"x", "p", "lam:f", "lam:g"},
                      {"triu:hess:gamma:x:x", "triu:hess:gamma:p:p"}, {{"gamma", {"f", "g"}}});
    }
    hesslag_sp_ = get_function("nlp_hess_l").sparsity_out(0);
    hesslagp_sp_ = get_function("nlp_hess_l").sparsity_out(1);
    casadi_assert(hesslag_sp_.is_triu(), "Hessian must be upper triangular");
    if (convexify_strategy!="none") {
      convexify_ = true;
      Dict opts;
      opts["strategy"] = convexify_strategy;
      opts["margin"] = convexify_margin;
      opts["max_iter_eig"] = max_iter_eig;
      opts["verbose"] = verbose_;
      hesslag_sp_ = Convexify::setup(convexify_data_, hesslag_sp_, opts);
    }
  } else if (pass_nonlinear_variables_) {
    nl_ex_ = oracle_.which_depends("x", {"f", "g"}, 2, false);
  }

  // Allocate work vectors
  alloc_w(ng_, true); // gk_
  alloc_w(nx_, true); // grad_fk_
  alloc_w(jacg_sp_.nnz(), true); // jac_gk_
  if (exact_hessian_) {
    alloc_w(hesslag_sp_.nnz(), true); // hess_lk_
  }
  if (convexify_) {
    alloc_iw(convexify_data_.sz_iw);
    alloc_w(convexify_data_.sz_w);
  }
}

int SIpoptInterface::init_mem(void* mem) const {
  if (Nlpsol::init_mem(mem)) return 1;
  auto m = static_cast<SIpoptMemory*>(mem);

  // Start an IPOPT application
  Ipopt::SmartPtr<Ipopt::IpoptApplication> *app = new Ipopt::SmartPtr<Ipopt::IpoptApplication>();
  m->app = static_cast<void*>(app);
  *app = new Ipopt::IpoptApplication(false);

  if(run_sens_) {
    Ipopt::SmartPtr<Ipopt::SensApplication> *app_sens = new Ipopt::SmartPtr<Ipopt::SensApplication>();
    m->app_sens = static_cast<void*>(app_sens);
    *app_sens = new Ipopt::SensApplication(
        (*app)->Jnlst(), (*app)->Options(), (*app)->RegOptions());

    // Register sIPOPT options
    RegisterOptions_sIPOPT((*app)->RegOptions());
    (*app)->Options()->SetRegisteredOptions((*app)->RegOptions());
  }

  // Direct output through casadi::uout()
  StreamJournal* jrnl_raw = new StreamJournal("console", J_ITERSUMMARY);
  jrnl_raw->SetOutputStream(&casadi::uout());
  jrnl_raw->SetPrintLevel(J_DBG, J_NONE);
  SmartPtr<Journal> jrnl = jrnl_raw;
  (*app)->Jnlst()->AddJournal(jrnl);

  // Create an Ipopt user class -- need to use Ipopts spart pointer class
  Ipopt::SmartPtr<Ipopt::TNLP> *userclass = new Ipopt::SmartPtr<Ipopt::TNLP>();
  m->userclass = static_cast<void*>(userclass);
  *userclass = new SIpoptUserClass(*this, m);

  if (verbose_) {
    uout() << "There are " << nx_ << " variables and " << ng_ << " constraints." << std::endl;
    if (exact_hessian_) uout() << "Using exact Hessian" << std::endl;
    else             uout() << "Using limited memory Hessian approximation" << std::endl;
  }

  // Get all options available in (s)IPOPT
  auto regops = (*app)->RegOptions()->RegisteredOptionsList();

  Dict options = Options::sanitize(opts_);
  // Replace resto group with prefixes
  auto it = options.find("resto");
  if (it!=options.end()) {
    Dict resto_options = it->second;
    options.erase(it);
    for (auto&& op : resto_options) {
      options["resto." + op.first] = op.second;
    }
  }

  // Pass all the options to ipopt
  for (auto&& op : options) {

    // There might be options with a resto prefix.
    std::string option_name = op.first;
    if (startswith(option_name, "resto.")) {
      option_name = option_name.substr(6);
    }

    // Find the option
    auto regops_it = regops.find(option_name);
    if (regops_it==regops.end()) {
      casadi_error("No such IPOPT option: " + op.first);
    }

    // Get the type
    Ipopt::RegisteredOptionType ipopt_type = regops_it->second->Type();

    // Pass to IPOPT
    bool ret;
    switch (ipopt_type) {
      case Ipopt::OT_Number:
        ret = (*app)->Options()->SetNumericValue(op.first, op.second.to_double(), false);
        break;
      case Ipopt::OT_Integer:
        ret = (*app)->Options()->SetIntegerValue(op.first, op.second.to_int(), false);
        break;
      case Ipopt::OT_String:
        ret = (*app)->Options()->SetStringValue(op.first, op.second.to_string(), false);
        break;
      case Ipopt::OT_Unknown:
      default:
        casadi_warning("Cannot handle option \"" + op.first + "\", ignored");
        continue;
    }
    if (!ret) casadi_error("Invalid options were detected by Ipopt.");
  }

  // Override IPOPT's default linear solver
  if (opts_.find("linear_solver") == opts_.end()) {
    char * default_solver = getenv("IPOPT_DEFAULT_LINEAR_SOLVER");
    if (default_solver) {
      bool ret = (*app)->Options()->SetStringValue("linear_solver", default_solver, false);
      casadi_assert(ret, "Corrupted IPOPT_DEFAULT_LINEAR_SOLVER environmental variable");
    } else {
      // Fall back to MUMPS (avoid user issues after SPRAL was added to binaries and
      // chosen default by Ipopt)
      bool ret = (*app)->Options()->SetStringValue("linear_solver", "mumps", false);
      casadi_assert_dev(ret);
    }

  }

  // Intialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status = (*app)->Initialize();
  casadi_assert(status == Solve_Succeeded, "Error during IPOPT initialization");

  if(run_sens_) {
    Ipopt::SmartPtr<Ipopt::SensApplication> *app_sens =
        static_cast<Ipopt::SmartPtr<Ipopt::SensApplication>*>(m->app_sens);
    (*app_sens)->Initialize();
  }

  if (convexify_) m->add_stat("convexify");
  return 0;
}

void SIpoptInterface::set_work(void* mem, const double**& arg, double**& res,
                              casadi_int*& iw, double*& w) const {
  auto m = static_cast<SIpoptMemory*>(mem);

  // Set work in base classes
  Nlpsol::set_work(mem, arg, res, iw, w);

  // Work vectors
  m->gk = w; w += ng_;
  m->grad_fk = w; w += nx_;
  m->jac_gk = w; w += jacg_sp_.nnz();
  if (exact_hessian_) {
    m->hess_lk = w; w += hesslag_sp_.nnz();
  }
}

inline const char* return_status_string(Ipopt::ApplicationReturnStatus status) {
  switch (status) {
    case Solve_Succeeded:
      return "Solve_Succeeded";
    case Solved_To_Acceptable_Level:
      return "Solved_To_Acceptable_Level";
    case Infeasible_Problem_Detected:
      return "Infeasible_Problem_Detected";
    case Search_Direction_Becomes_Too_Small:
      return "Search_Direction_Becomes_Too_Small";
    case Diverging_Iterates:
      return "Diverging_Iterates";
    case User_Requested_Stop:
      return "User_Requested_Stop";
    case Maximum_Iterations_Exceeded:
      return "Maximum_Iterations_Exceeded";
    case Restoration_Failed:
      return "Restoration_Failed";
    case Error_In_Step_Computation:
      return "Error_In_Step_Computation";
    case Not_Enough_Degrees_Of_Freedom:
      return "Not_Enough_Degrees_Of_Freedom";
    case Invalid_Problem_Definition:
      return "Invalid_Problem_Definition";
    case Invalid_Option:
      return "Invalid_Option";
    case Invalid_Number_Detected:
      return "Invalid_Number_Detected";
    case Unrecoverable_Exception:
      return "Unrecoverable_Exception";
    case NonIpopt_Exception_Thrown:
      return "NonIpopt_Exception_Thrown";
    case Insufficient_Memory:
      return "Insufficient_Memory";
    case Internal_Error:
      return "Internal_Error";
    case Maximum_CpuTime_Exceeded:
      return "Maximum_CpuTime_Exceeded";
    case Feasible_Point_Found:
      return "Feasible_Point_Found";
    case Maximum_WallTime_Exceeded:
      return "Maximum_WallTime_Exceeded";
  }
  return "Unknown";
}

int SIpoptInterface::solve(void* mem) const {
  auto m = static_cast<SIpoptMemory*>(mem);
  auto d_nlp = &m->d_nlp;

  // Reset statistics
  m->inf_pr.clear();
  m->inf_du.clear();
  m->mu.clear();
  m->d_norm.clear();
  m->regularization_size.clear();
  m->alpha_pr.clear();
  m->alpha_du.clear();
  m->obj.clear();
  m->ls_trials.clear();

  // Reset number of iterations
  m->n_iter = 0;

  // Get back the smart pointers
  Ipopt::SmartPtr<Ipopt::TNLP> *userclass =
      static_cast<Ipopt::SmartPtr<Ipopt::TNLP>*>(m->userclass);
  Ipopt::SmartPtr<Ipopt::IpoptApplication> *app =
      static_cast<Ipopt::SmartPtr<Ipopt::IpoptApplication>*>(m->app);

  // Ask Ipopt to solve the problem
  Ipopt::ApplicationReturnStatus status = (*app)->OptimizeTNLP(*userclass);
  m->return_status = return_status_string(status);
  m->success = status==Solve_Succeeded || status==Solved_To_Acceptable_Level
               || status==Feasible_Point_Found;
  if (status==Maximum_Iterations_Exceeded ||
      status==Maximum_WallTime_Exceeded ||
      status==Maximum_CpuTime_Exceeded) m->unified_return_status = SOLVER_RET_LIMITED;

  // Save results to outputs
  casadi_copy(m->gk, ng_, d_nlp->z + nx_);

  if (clip_inactive_lam_) {
    // Compute a margin
    double margin;
    if (inactive_lam_strategy_=="abstol") {
      margin = inactive_lam_value_;
    } else if (inactive_lam_strategy_=="reltol") {
      double constr_viol_tol;
      (*app)->Options()->GetNumericValue("constr_viol_tol", constr_viol_tol, "");
      if (status==Solved_To_Acceptable_Level) {
        (*app)->Options()->GetNumericValue("acceptable_constr_viol_tol", constr_viol_tol, "");
      }
      margin = inactive_lam_value_*constr_viol_tol;
    } else {
      casadi_error("inactive_lam_strategy '" + inactive_lam_strategy_ +
                   "' unknown. Use 'abstol' or reltol'.");
    }

    for (casadi_int i=0; i<nx_ + ng_; ++i) {
      // Sufficiently inactive -> make multiplier exactly zero
      if (d_nlp->lam[i]>0 && d_nlp->ubz[i] - d_nlp->z[i] > margin) d_nlp->lam[i]=0;
      if (d_nlp->lam[i]<0 && d_nlp->z[i] - d_nlp->lbz[i] > margin) d_nlp->lam[i]=0;
    }
  }

  if(m->success && run_sens_) {
    Ipopt::SmartPtr<Ipopt::SensApplication> *app_sens =
        static_cast<Ipopt::SmartPtr<Ipopt::SensApplication>*>(m->app_sens);
    // give pointers to Ipopt algorithm objects to Sens Application
    (*app_sens)->SetIpoptAlgorithmObjects(*app, status);
    (*app_sens)->Run();
  }

  return 0;
}

bool SIpoptInterface::
intermediate_callback(SIpoptMemory* m, const double* x, const double* z_L, const double* z_U,
                      const double* g, const double* lambda, double obj_value, int iter,
                      double inf_pr, double inf_du, double mu, double d_norm,
                      double regularization_size, double alpha_du, double alpha_pr,
                      int ls_trials, bool full_callback) const {
  auto d_nlp = &m->d_nlp;
  m->n_iter += 1;
  try {
    m->inf_pr.push_back(inf_pr);
    m->inf_du.push_back(inf_du);
    m->mu.push_back(mu);
    m->d_norm.push_back(d_norm);
    m->regularization_size.push_back(regularization_size);
    m->alpha_pr.push_back(alpha_pr);
    m->alpha_du.push_back(alpha_du);
    m->ls_trials.push_back(ls_trials);
    m->obj.push_back(obj_value);
    if (!fcallback_.is_null()) {
      ScopedTiming tic(m->fstats.at("callback_fun"));
      if (full_callback) {
        casadi_copy(x, nx_, d_nlp->z);
        for (casadi_int i=0; i<nx_; ++i) {
          d_nlp->lam[i] = z_U[i]-z_L[i];
        }
        casadi_copy(lambda, ng_, d_nlp->lam + nx_);
//          casadi_copy(lambda + ng_, np_, d_nlp->lam_p);
        casadi_copy(g, ng_, m->gk);
      } else {
        if (iter==0) {
          uerr()
              << "Warning: intermediate_callback is disfunctional in your installation. "
                 "You will only be able to use stats(). "
                 "See https://github.com/casadi/casadi/wiki/enableIpoptCallback to enable it."
              << std::endl;
        }
      }

      // Inputs
      std::fill_n(m->arg, fcallback_.n_in(), nullptr);
      if (full_callback) {
        // The values used below are meaningless
        // when not doing a full_callback
        m->arg[NLPSOL_X] = x;
        m->arg[NLPSOL_F] = &obj_value;
        m->arg[NLPSOL_G] = g;
        m->arg[NLPSOL_LAM_P] = nullptr;
        m->arg[NLPSOL_LAM_X] = d_nlp->lam;
        m->arg[NLPSOL_LAM_G] = d_nlp->lam + nx_;
      }

      // Outputs
      std::fill_n(m->res, fcallback_.n_out(), nullptr);
      double ret_double;
      m->res[0] = &ret_double;

      fcallback_(m->arg, m->res, m->iw, m->w, 0);
      casadi_int ret = static_cast<casadi_int>(ret_double);

      return  !ret;
    } else {
      return 1;
    }

  } catch(KeyboardInterruptException& ex) {
    return 0;
  } catch(std::exception& ex) {
    casadi_warning("intermediate_callback: " + std::string(ex.what()));
    if (iteration_callback_ignore_errors_) return 1;
    return 0;
  }
}

void SIpoptInterface::
finalize_solution(SIpoptMemory* m, const double* x, const double* z_L, const double* z_U,
                  const double* g, const double* lambda, double obj_value,
                  int iter_count, const std::vector<double> &perturbed_x) const {
  auto d_nlp = &m->d_nlp;
  try {
    // Get primal solution
    casadi_copy(x, nx_, d_nlp->z);

    // Get optimal cost
    d_nlp->objective = obj_value;

    // Get dual solution (simple bounds)
    for (casadi_int i=0; i<nx_; ++i) {
      d_nlp->lam[i] = z_U[i]-z_L[i];
    }

    // Get dual solution (nonlinear bounds)
    casadi_copy(lambda, ng_, d_nlp->lam + nx_);

    // Get the constraints
    casadi_copy(g, ng_, m->gk);

    // Get statistics
    m->iter_count = iter_count;

    m->perturbed_x = perturbed_x;

  } catch(std::exception& ex) {
    uerr() << "finalize_solution failed: " << ex.what() << std::endl;
  }
}

bool SIpoptInterface::
get_bounds_info(SIpoptMemory* m, double* x_l, double* x_u,
                double* g_l, double* g_u) const {
  auto d_nlp = &m->d_nlp;
  try {
    casadi_copy(d_nlp->lbz, nx_, x_l);
    casadi_copy(d_nlp->ubz, nx_, x_u);
    casadi_fill(x_l + nx_, np_, -inf);
    casadi_fill(x_u + nx_, np_, inf);
    casadi_copy(d_nlp->lbz+nx_, ng_, g_l);
    casadi_copy(d_nlp->ubz+nx_, ng_, g_u);
    casadi_copy(d_nlp->p, np_, g_l+ng_);
    casadi_copy(d_nlp->p, np_, g_u+ng_);
    return true;
  } catch(std::exception& ex) {
    uerr() << "get_bounds_info failed: " << ex.what() << std::endl;
    return false;
  }
}

bool SIpoptInterface::
get_starting_point(SIpoptMemory* m, bool init_x, double* x,
                   bool init_z, double* z_L, double* z_U,
                   bool init_lambda, double* lambda) const {
  auto d_nlp = &m->d_nlp;
  try {
    // Initialize primal variables
    if (init_x) {
      casadi_copy(d_nlp->z, nx_, x);
      casadi_copy(d_nlp->p, np_, x+nx_);
    }

    // Initialize dual variables (simple bounds)
    if (init_z) {
      for (casadi_int i=0; i<nx_; ++i) {
        z_L[i] = std::max(0., -d_nlp->lam[i]);
        z_U[i] = std::max(0., d_nlp->lam[i]);
      }
    }

    // Initialize dual variables (nonlinear bounds)
    if (init_lambda) {
      casadi_copy(d_nlp->lam + nx_, ng_, lambda);
    }

    return true;
  } catch(std::exception& ex) {
    uerr() << "get_starting_point failed: " << ex.what() << std::endl;
    return false;
  }
}

void SIpoptInterface::get_nlp_info(SIpoptMemory* m, int& nx, int& ng,
                                  int& nnz_jac_g, int& nnz_h_lag) const {
  try {
    // Number of variables
    nx = nx_ + np_;

    // Number of constraints
    ng = ng_ + np_;

    // Number of Jacobian nonzeros
    nnz_jac_g = (ng_==0 ? 0 : jacg_sp_.nnz()) + jacgp_sp_.nnz() + np_;

    // Number of Hessian nonzeros (only upper triangular half)
    nnz_h_lag = exact_hessian_ ? hesslag_sp_.nnz() + hesslagp_sp_.nnz() : 0;

  } catch(std::exception& ex) {
    uerr() << "get_nlp_info failed: " << ex.what() << std::endl;
  }
}

int SIpoptInterface::get_number_of_nonlinear_variables() const {
  try {
    if (!pass_nonlinear_variables_) {
      // No Hessian has been interfaced
      return -1;
    } else {
      // Number of variables that appear nonlinearily
      int nv = 0;
      for (auto&& i : nl_ex_) if (i) nv++;
      return nv;
    }
  } catch(std::exception& ex) {
    uerr() << "get_number_of_nonlinear_variables failed: " << ex.what() << std::endl;
    return -1;
  }
}

bool SIpoptInterface::
get_list_of_nonlinear_variables(int num_nonlin_vars, int* pos_nonlin_vars) const {
  try {
    for (int i=0; i<nl_ex_.size(); ++i) {
      if (nl_ex_[i]) *pos_nonlin_vars++ = i;
    }
    return true;
  } catch(std::exception& ex) {
    uerr() << "get_list_of_nonlinear_variables failed: " << ex.what() << std::endl;
    return false;
  }
}

bool SIpoptInterface::
get_var_con_metadata(SIpoptMemory *m, std::map<std::string, std::vector<std::string> >& var_string_md,
                     std::map<std::string, std::vector<int> >& var_integer_md,
                     std::map<std::string, std::vector<double> >& var_numeric_md,
                     std::map<std::string, std::vector<std::string> >& con_string_md,
                     std::map<std::string, std::vector<int> >& con_integer_md,
                     std::map<std::string, std::vector<double> >& con_numeric_md) const {
  for (auto&& op : var_string_md_) var_string_md[op.first] = op.second;
  for (auto&& op : var_integer_md_) var_integer_md[op.first] = op.second;
  for (auto&& op : var_numeric_md_) var_numeric_md[op.first] = op.second;
  for (auto&& op : con_string_md_) con_string_md[op.first] = op.second;
  for (auto&& op : con_integer_md_) con_integer_md[op.first] = op.second;
  for (auto&& op : con_numeric_md_) con_numeric_md[op.first] = op.second;

  if(run_sens_) {
    con_integer_md["sens_init_constr"] = std::vector<int>(ng_ + np_, 0);
    var_integer_md["sens_state_1"] = std::vector<int>(nx_ + np_, 0);
    var_numeric_md["sens_state_value_1"] = std::vector<double>(nx_ + np_, 0.0);

    for(int i = 0; i < np_; i++) {
      con_integer_md["sens_init_constr"][ng_ + i] = i+1;
      var_integer_md["sens_state_1"][nx_ + i] = i+1;
      var_numeric_md["sens_state_value_1"][nx_ + i] = perturbed_p_.empty() ? m->d_nlp.p[i] : perturbed_p_[i];
    }
  }
  return true;
}

SIpoptMemory::SIpoptMemory() {
  this->app = nullptr;
  this->app_sens = nullptr;
  this->userclass = nullptr;
  this->return_status = "Unset";
}

SIpoptMemory::~SIpoptMemory() {
  // Free Ipopt application instance (or rather, the smart pointer holding it)
  if (this->app != nullptr) {
    delete static_cast<Ipopt::SmartPtr<Ipopt::IpoptApplication>*>(this->app);
  }

  // Free Ipopt sens application instance (or rather, the smart pointer holding it)
  if(this->app_sens != nullptr) {
    delete static_cast<Ipopt::SmartPtr<Ipopt::IpoptApplication>*>(this->app_sens);
  }

  // Free Ipopt user class (or rather, the smart pointer holding it)
  if (this->userclass != nullptr) {
    delete static_cast<Ipopt::SmartPtr<Ipopt::TNLP>*>(this->userclass);
  }
}

Dict SIpoptInterface::get_stats(void* mem) const {
  Dict stats = Nlpsol::get_stats(mem);
  auto m = static_cast<SIpoptMemory*>(mem);
  stats["return_status"] = m->return_status;
  stats["iter_count"] = m->iter_count;
  if (!m->inf_pr.empty()) {
    Dict iterations;
    iterations["inf_pr"] = m->inf_pr;
    iterations["inf_du"] = m->inf_du;
    iterations["mu"] = m->mu;
    iterations["d_norm"] = m->d_norm;
    iterations["regularization_size"] = m->regularization_size;
    iterations["obj"] = m->obj;
    iterations["alpha_pr"] = m->alpha_pr;
    iterations["alpha_du"] = m->alpha_du;
    stats["iterations"] = iterations;
  }
  if (run_sens_) {
    stats["perturbed_x"] = m->perturbed_x;

    Ipopt::SmartPtr<Ipopt::SensApplication> *app_sens =
        static_cast<Ipopt::SmartPtr<Ipopt::SensApplication>*>(m->app_sens);

    auto compute_dsdp = opts_.find("compute_dsdp");
    if(compute_dsdp != opts_.end() && compute_dsdp->second.as_string() == "yes") {
      casadi_int np = (*app_sens)->np();
      if(np < 0) {
        stats["sens_x"] = std::vector<double>();
        stats["sens_l"] = std::vector<double>();
        stats["sens_zl"] = std::vector<double>();
        stats["sens_zu"] = std::vector<double>();
      } else {
        std::vector<double> sens_x((*app_sens)->nx() * np), sens_l((*app_sens)->nl() * np),
            sens_zl((*app_sens)->nzl() * np), sens_zu((*app_sens)->nzu() * np);
        (*app_sens)->GetSensitivityMatrix(
            sens_x.data(), sens_l.data(), sens_zl.data(), sens_zu.data());
        stats["sens_x"] = sens_x;
        stats["sens_l"] = sens_l;
        stats["sens_zl"] = sens_zl;
        stats["sens_zu"] = sens_zu;
      }
    }
  }
  return stats;
}

SIpoptInterface::SIpoptInterface(DeserializingStream& s) : Nlpsol(s) {
  int version = s.version("SIpoptInterface", 1, 3);
  s.unpack("SIpoptInterface::jacg_sp", jacg_sp_);
  s.unpack("SIpoptInterface::hesslag_sp", hesslag_sp_);
  s.unpack("SIpoptInterface::exact_hessian", exact_hessian_);
  s.unpack("SIpoptInterface::opts", opts_);
  s.unpack("SIpoptInterface::pass_nonlinear_variables", pass_nonlinear_variables_);
  s.unpack("SIpoptInterface::nl_ex", nl_ex_);
  s.unpack("SIpoptInterface::var_string_md", var_string_md_);
  s.unpack("SIpoptInterface::var_integer_md", var_integer_md_);
  s.unpack("SIpoptInterface::var_numeric_md", var_numeric_md_);
  s.unpack("SIpoptInterface::con_string_md", con_string_md_);
  s.unpack("SIpoptInterface::con_integer_md", con_integer_md_);
  s.unpack("SIpoptInterface::con_numeric_md", con_numeric_md_);
  if (version>=2) {
    s.unpack("SIpoptInterface::convexify", convexify_);
    if (convexify_) Convexify::deserialize(s, "SIpoptInterface::", convexify_data_);
  }

  if (version>=3) {
    s.unpack("SIpoptInterface::clip_inactive_lam", clip_inactive_lam_);
    s.unpack("SIpoptInterface::inactive_lam_strategy", inactive_lam_strategy_);
    s.unpack("SIpoptInterface::inactive_lam_value", inactive_lam_value_);
  } else {
    clip_inactive_lam_ = false;
    inactive_lam_strategy_ = "reltol";
    inactive_lam_value_ = 10;
  }
}

void SIpoptInterface::serialize_body(SerializingStream &s) const {
  Nlpsol::serialize_body(s);
  s.version("SIpoptInterface", 3);
  s.pack("SIpoptInterface::jacg_sp", jacg_sp_);
  s.pack("SIpoptInterface::hesslag_sp", hesslag_sp_);
  s.pack("SIpoptInterface::exact_hessian", exact_hessian_);
  s.pack("SIpoptInterface::opts", opts_);
  s.pack("SIpoptInterface::pass_nonlinear_variables", pass_nonlinear_variables_);
  s.pack("SIpoptInterface::nl_ex", nl_ex_);
  s.pack("SIpoptInterface::var_string_md", var_string_md_);
  s.pack("SIpoptInterface::var_integer_md", var_integer_md_);
  s.pack("SIpoptInterface::var_numeric_md", var_numeric_md_);
  s.pack("SIpoptInterface::con_string_md", con_string_md_);
  s.pack("SIpoptInterface::con_integer_md", con_integer_md_);
  s.pack("SIpoptInterface::con_numeric_md", con_numeric_md_);
  s.pack("SIpoptInterface::convexify", convexify_);
  if (convexify_) Convexify::serialize(s, "SIpoptInterface::", convexify_data_);

  s.pack("SIpoptInterface::clip_inactive_lam", clip_inactive_lam_);
  s.pack("SIpoptInterface::inactive_lam_strategy", inactive_lam_strategy_);
  s.pack("SIpoptInterface::inactive_lam_value", inactive_lam_value_);

}

} // namespace casadi
