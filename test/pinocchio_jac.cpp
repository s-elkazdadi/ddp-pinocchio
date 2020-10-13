#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

auto main() -> int {
  pinocchio::Model model;
  pinocchio::GeometryModel geom;
  fmt::string_view home_dir = std::getenv("HOME"); // TODO: windows?
  pinocchio::urdf::buildModel(
      std::string{home_dir.begin(), home_dir.end()} +
          "/pinocchio/models/others/robots/ur_description/urdf/ur5_gripper.urdf",
      model);

  pinocchio::Data data(model);
  pinocchio::Data data2(model);

  using vec = Eigen::VectorXd;
  using mat = Eigen::MatrixXd;

  auto nq = model.nq;
  auto nv = model.nv;
  vec q{nq};
  pinocchio::neutral(model, q);
  std::cout << q << '\n';

  vec v = 1e-4 * vec::Random(nv);

  auto q2 = pinocchio::integrate(model, q, v);

  pinocchio::computeJointJacobians(model, data, q);
  mat jac{6, nv};

  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::framesForwardKinematics(model, data2, q2);

  for (size_t i = 0; i < data.oMf.size(); ++i) {
    jac.setZero();
    pinocchio::getFrameJacobian(model, data, i, pinocchio::LOCAL_WORLD_ALIGNED, jac);

    fmt::print("jac         : {}\n", (data2.oMf[i].translation() - data.oMf[i].translation()).transpose());
    fmt::print("finite diff : {}\n", (jac.topRows<3>() * v).transpose());
  }
}
