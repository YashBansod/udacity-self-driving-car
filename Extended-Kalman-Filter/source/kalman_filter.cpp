#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /**
     * TODO: predict the state
     */
    x_ = F_ * x_;
    MatrixXd F_t = F_.transpose();
    P_ = F_ * P_ * F_t + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
     * TODO: update the state by using Kalman Filter equations
     */

    VectorXd y = z - H_ * x_;
    MatrixXd H_t = H_.transpose();
    MatrixXd S = H_ * P_ * H_t + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * H_t * Si;

    x_ = x_ + (K * y);
    MatrixXd I_ = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
     * TODO: update the state by using Extended Kalman Filter equations
     */
    Eigen::MatrixXd I_;
    I_ = MatrixXd::Identity(4, 4);

    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);

    double h_1 = sqrt(px * px + py * py);
    double h_2 = atan2(py, px);
    double h_3;

    if (fabs(h_1) > 0.0001) {
        h_3 = ((px * vx) + (py * vy)) / h_1;
    }
    else {
        h_3 = 0;
    }

    VectorXd h_x(3);
    h_x << h_1, h_2, h_3;

    VectorXd y = z - h_x;
    y[1] = fmod(y[1] + M_PI, M_PI*2) - M_PI;

    MatrixXd Hj_t = H_.transpose();
    MatrixXd S = H_ * P_ * Hj_t + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * Hj_t * Si;

    x_ = x_ + (K * y);
    P_ = (I_ - K * H_) * P_;
}
