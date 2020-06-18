#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    /**
     * TODO: Finish initializing the FusionEKF.
     * TODO: Set the process and measurement noises
     */

    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    Hj_ << 1, 1, 0, 0,
            1, 1, 0, 0,
            1, 1, 1, 1;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    /**
     * Initialization
     */
    if (!is_initialized_) {
        /**
         * TODO: Initialize the state ekf_.x_ with the first measurement.
         * TODO: Create the covariance matrix.
         * You'll need to convert radar from polar to cartesian coordinates.
         */

        // first measurement
        cout << "EKF: " << endl;
        VectorXd x_state(4);
        MatrixXd P_cov(4, 4);
        MatrixXd F_state_trans(4, 4);
        MatrixXd Q_proc_cov(4, 4);

        F_state_trans << 1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;

        Q_proc_cov << 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // TODO: Convert radar from polar to cartesian coordinates
            //         and initialize state.

            double rho = measurement_pack.raw_measurements_[0];     // range
            double phi = measurement_pack.raw_measurements_[1];     // bearing
            double rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
            double cos_phi = cos(phi);
            double sin_phi = sin(phi);
            x_state << rho * cos_phi, rho * sin_phi, rho_dot * cos_phi, rho_dot * sin_phi;


            P_cov << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 10, 0,
                    0, 0, 0, 10;

            ekf_.Init(x_state, P_cov, F_state_trans, Hj_, R_radar_, Q_proc_cov);

        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // TODO: Initialize state.
            x_state << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;

            P_cov << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1000, 0,
                    0, 0, 0, 1000;

            ekf_.Init(x_state, P_cov, F_state_trans, H_laser_, R_laser_, Q_proc_cov);
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /**
     * Prediction
     */

    /**
     * TODO: Update the state transition matrix F according to the new elapsed time.
     * Time is measured in seconds.
     * TODO: Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    double dt_2 = dt * dt;
    double dt_3_2 = (dt_2 * dt) / 2.0;
    double dt_4_4 = (dt_2 * dt_2) / 4.0;

    //set the acceleration noise components
    double noise_ax = 9.0;
    double noise_ay = 9.0;

    //set the state transition matrix F
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    //set the process covariance matrix Q
    ekf_.Q_(0, 0) = dt_4_4 * noise_ax;
    ekf_.Q_(0, 2) = dt_3_2 * noise_ax;
    ekf_.Q_(1, 1) = dt_4_4 * noise_ay;
    ekf_.Q_(1, 3) = dt_3_2 * noise_ay;

    ekf_.Q_(2, 0) = dt_3_2 * noise_ax;
    ekf_.Q_(2, 2) = dt_2 * noise_ax;
    ekf_.Q_(3, 1) = dt_3_2 * noise_ay;
    ekf_.Q_(3, 3) = dt_2 * noise_ay;

    ekf_.Predict();

    /**
     * Update
     */

    /**
     * TODO:
     * - Use the sensor type to perform the update step.
     * - Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // TODO: Radar updates
        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);

    } else {
        // TODO: Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
