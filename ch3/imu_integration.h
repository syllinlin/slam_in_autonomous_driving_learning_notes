//
// Created by xiang on 2021/11/5.
//

#ifndef SLAM_IN_AUTO_DRIVING_IMU_INTEGRATION_H
#define SLAM_IN_AUTO_DRIVING_IMU_INTEGRATION_H

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/nav_state.h"
#include<vector>

namespace sad {

/**
 * 本程序演示单纯靠IMU的积分
 */
class IMUIntegration {
   public:
    IMUIntegration(const Vec3d& gravity, const Vec3d& init_bg, const Vec3d& init_ba)
        : gravity_(gravity), bg_(init_bg), ba_(init_ba) {}

    // 增加imu读数
    void AddIMU(const IMU& imu) {
        double dt = imu.timestamp_ - timestamp_;
        if (dt > 0 && dt < 0.1) { // 要求频率大于10Hz
            // 假设IMU时间间隔在0至0.1以内
            // 测试一下PoseByIMUOnQuat和正常计算的R_相差大不大
            PoseByIMUOnQuat(imu);
            nor_R_ = nor_R_ * Sophus::SO3d::exp((imu.gyro_ - bg_) * dt);
            std::cout << "----------------------" << std::endl;
            std::cout << "nor_R = " << nor_R_.matrix() << std::endl;
            std::cout << "R_ = " << R_.matrix() << std::endl; 
            // R_ = R_ * Sophus::SO3d::exp((imu.gyro_ - bg_) * dt);
            p_ = p_ + v_ * dt + 0.5 * gravity_ * dt * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt;
            v_ = v_ + R_ * (imu.acce_ - ba_) * dt + gravity_ * dt;
        }

        // 更新时间
        timestamp_ = imu.timestamp_;
    }

    // 重力初始化，将Z轴朝向转为(0, 0, -1)作为世界坐标系
    // 打算采用简单处理方法：取前15个IMU加速度测量值进行累加平均作为重力加速度的初始值和(0, 0, -1)之间的旋转作为初始值
    bool init_success = true;
    std::vector<IMU> imu_vec_;
    void InitState() {
        if(imu_vec_.size() < 15) return;
        init_success = true;
        Vec3d aver_acc(0.0, 0.0, 0.0);
        for(size_t i=0; i<imu_vec_.size(); ++i) {
            aver_acc += imu_vec_[i].acce_;
        }
        aver_acc /= imu_vec_.size();
        //std::cout << "aver_acc = " << aver_acc << ", numbers = " << imu_vec_.size() << std::endl;

        Vec3d ng1 = aver_acc.normalized();
        Vec3d ng2 = Vec3d(0.0, 0.0, -1.0);
        Mat3d R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        R_ = SO3::fitToSO3(R0);
        //std::cout << "ng1 = " << ng1 << std::endl;
        //std::cout << "R0 = " << R0 << std::endl;
        //std::cout << "R_ = " << R_.matrix() << std::endl;
    }

    Quatd last_q;
    Vec4d last_q_vec;
    Vec7d x_imupose;
    Mat7d F_imupose;
    Mat7d P_imupose;
    Mat7d Q_imupose;
    Mat3d R_imupose; // 三轴加速度测量噪声
    Eigen::Matrix<double, 3, 7> H_imupose;
    Eigen::Matrix<double, 7, 3> K_imupose;
    Vec3d z_imupose;
    void InitStateForIMUPose() {
        Vec4d q0(1.0, 0.0, 0.0, 0.0);
        x_imupose << q0, bg_;
        F_imupose = Mat7d::Identity();
        P_imupose.setZero();
        Q_imupose.setZero(); // 因为没有确切的数据，所以我直接赋值为0.001
        Q_imupose(4, 4) = Q_imupose(5, 5) = Q_imupose(6, 6) = 0.0001;
        R_imupose = 0.001*Mat3d::Identity();
    }
    
    // 先用基于四元数的表达实现基于EKF的IMU位姿估计
    void PoseByIMUOnQuat(const IMU& imu) {
        double dt = imu.timestamp_ - timestamp_;
        if (dt < 0 && dt > 0.1) return;
        last_q = R_.unit_quaternion(); // 获取R_对应的单位四元数
        last_q_vec = Vec4d(last_q.w(), last_q.x(), last_q.y(), last_q.z());
        x_imupose.head(4) = last_q_vec;

        // 1、预测当前cur_q_vec
        Mat4d Omega = 0.5*dt*LeftMatrix(imu.gyro_, x_imupose.tail(3));
        // Vec4d cur_q_vec = last_q_vec + Omega*last_q_vec;    
        F_imupose.setZero();
        F_imupose.block<4, 4>(0, 0) = Omega + Mat4d::Identity();
        F_imupose.block<3, 3>(4, 4) = Mat3d::Identity();
        F_imupose.block<4, 3>(0, 4) = -0.5*dt*LeftQuatd(last_q_vec);
        x_imupose = F_imupose*x_imupose;

        // 2、P_imupose在预测阶段的更新
        P_imupose = F_imupose*P_imupose*(F_imupose).transpose() + Q_imupose;
        std::cout << "pred P = \n" << P_imupose << std::endl; 

        // 3、更新阶段中增益的计算
        double q1 = x_imupose[0], q2 = x_imupose[1], q3 = x_imupose[2], q4 = x_imupose[3];
        H_imupose.setZero();
        Eigen::Matrix<double, 3, 4> leftM34d;
        leftM34d << q3, -q4, q1, -q2,
                                  -q2, -q1, -q4, -q3,
                                  -q1, q2, q3, -q4;
        H_imupose.block<3, 4>(0, 0) = 2*leftM34d;
        Eigen::Matrix<double, 7, 3> PH = P_imupose*H_imupose.transpose();
        K_imupose = PH*(H_imupose*PH + R_imupose).inverse();

        // 4、更新阶段中x_imupose的更新
        z_imupose << -2*(q2*q4-q1*q3), -2*(q1*q2+q3*q4), -(q1*q1-q2*q2-q3*q3+q4*q4);
        // ！！！z_imupose和x_imupose是否需要在这里归一化，或者所有结果完了之后归一化？？？
        //z_imupose.normalize();，
        //x_imupose.head(4).normalize();
        x_imupose = x_imupose + K_imupose*(z_imupose-H_imupose*x_imupose);

        // 5、更新阶段中P_imupose的更新
        P_imupose = (Mat7d::Identity() - K_imupose*H_imupose)*P_imupose;

        // 将四元数的结果转到SO3形式
        Quatd q(x_imupose[0], x_imupose[1], x_imupose[2], x_imupose[3]);
        q.normalize();
        //R_ = SO3::exp(q.coeffs());
        R_ = SO3(q);
    }

    Mat4d LeftMatrix(const Vec3d& gyro, const Vec3d& bg_tmp) {
        Mat4d mat;
        mat << 0, -gyro.x()+bg_tmp.x(), -gyro.y()+bg_tmp.y(), -gyro.z()+bg_tmp.z(),
                       gyro.x()-bg_tmp.x(), 0, gyro.z()-bg_tmp.z(), -gyro.y()+bg_tmp.y(),
                       gyro.y()-bg_tmp.y(), -gyro.z()+bg_tmp.z(), 0, gyro.x()-bg_tmp.x(),
                       gyro.z()-bg_tmp.z(), gyro.y()-bg_tmp.y(), -gyro.x()+bg_tmp.x(), 0;
        return mat;
    }

    Eigen::Matrix<double, 4, 3> LeftQuatd(const Vec4d& q_vec) {
        Eigen::Matrix<double, 4, 3> res;
        res << -q_vec[1], -q_vec[2], -q_vec[3],
                     q_vec[0], -q_vec[3], q_vec[2],
                     q_vec[3], q_vec[0], -q_vec[1],
                     -q_vec[2], q_vec[1], q_vec[0];
        return res;
    }

    /// 组成NavState
    NavStated GetNavState() const { return NavStated(timestamp_, R_, p_, v_, bg_, ba_); }

    SO3 GetR() const { return R_; }
    Vec3d GetV() const { return v_; }
    Vec3d GetP() const { return p_; }

   private:
    // 累计量
    SO3 R_;
    // Shuyue Lin 
    SO3 nor_R_; // for test
    Vec3d v_ = Vec3d::Zero();
    Vec3d p_ = Vec3d::Zero();

    double timestamp_ = 0.0;

    // 零偏，由外部设定
    Vec3d bg_ = Vec3d::Zero();
    Vec3d ba_ = Vec3d::Zero();

    Vec3d gravity_ = Vec3d(0, 0, -9.8);  // 重力
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_IMU_INTEGRATION_H
