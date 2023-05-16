#include <casadi/casadi.hpp>

using namespace casadi;

int main() {
  auto x = SX::sym("x", 3);
  auto p = SX::sym("p", 2);
  SXDict nlp = {{"x", x},
                {"p", p},
                {"f", sumsqr(x)},
                {"g", SX::vertcat({
                                      6 * x(0) + 3 * x(1) + 2 * x(2) - p(0),
                                      p(1) * x(0) + x(1) - x(2) - 1,
                                  })}};
  Dict opts, solver_opts;
  solver_opts["sens_boundcheck"] = "yes";
  solver_opts["linear_solver"] = "ma27";
  solver_opts["run_sens"] = "yes";
  solver_opts["n_sens_steps"] = 1;
  solver_opts["compute_dsdp"] = "yes";
  solver_opts["fixed_variable_treatment"] = "relax_bounds";
//  solver_opts["hessian_approximation"] = "limited-memory";
  opts["sipopt"] = solver_opts;
  opts["perturbed_p"] = std::vector<double>{4.5, 1.0};
  auto solver = nlpsol("solver", "sipopt", nlp, opts);
  DMDict solver_in;
  solver_in["lbx"] = 0.0;
  solver_in["ubx"] = inf;
  solver_in["lbg"] = 0;
  solver_in["ubg"] = 0;
  solver_in["p"] = {5.0, 1.0};
  auto solver_out = solver(solver_in);
  std::cout << "solution: " << solver_out["x"] << std::endl;
  Dict stats = solver.stats();
  std::cout << "sens_x: " << stats["sens_x"].as_double_vector() << std::endl;
  std::cout << "perturbed_x: " << stats["perturbed_x"].as_double_vector() << std::endl;
  return 0;
}
