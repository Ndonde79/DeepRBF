#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stdexcept>
#include <cmath>
#include <vector>
namespace deeprbf {
// Projects a symmetric coregionalization matrix onto the PSD cone.
// This is a numerical safeguard, not a substitute for fitting a valid LMC.
inline Eigen::MatrixXd project_coregionalization_psd(const Eigen::MatrixXd& input, double floor=0.0) {
  if(input.rows()==0 || input.rows()!=input.cols() || !input.allFinite() || !std::isfinite(floor) || floor<0)
    throw std::invalid_argument("invalid coregionalization matrix or eigenvalue floor");
  const Eigen::MatrixXd symmetric = 0.5*(input+input.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  if(solver.info()!=Eigen::Success) throw std::runtime_error("coregionalization eigensolver failed");
  return solver.eigenvectors()*solver.eigenvalues().cwiseMax(floor).asDiagonal()*solver.eigenvectors().transpose();
}
inline bool is_coregionalization_psd(const Eigen::MatrixXd& input, double tolerance=1e-10) {
  if(input.rows()==0 || input.rows()!=input.cols() || !input.allFinite() || tolerance<0 || !std::isfinite(tolerance)) return false;
  if(!input.isApprox(input.transpose(),tolerance)) return false;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(input,Eigen::EigenvaluesOnly);
  return solver.info()==Eigen::Success && solver.eigenvalues().minCoeff()>=-tolerance;
}
}
