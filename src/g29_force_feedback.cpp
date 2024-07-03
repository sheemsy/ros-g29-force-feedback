#include <ros/ros.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <dirent.h>
#include <string.h>
#include <std_msgs/Float64.h>

#include "ros_g29_force_feedback/ForceFeedback.h"

class G29ForceFeedback {

private:
    ros::Subscriber sub_target;
    ros::Subscriber sub_carla_steering;
    ros::Timer timer;

    // device info
    int m_device_handle;
    int m_axis_code = ABS_X;
    int m_axis_min;
    int m_axis_max;

    // rosparam
    std::string m_device_name = "";
    double m_loop_rate = 0.1;
    double m_max_torque = 1.0;
    double m_min_torque = 0.02;
    double m_brake_position = 0.2;
    double m_brake_torque_rate = 0.1;
    double m_auto_centering_max_torque = 0.3;
    double m_auto_centering_max_position = 0.2;
    double m_eps = 0.01;
    double m_min_feedback_torque = 0.1; // 최소 피드백 토크 증가
    double m_feedback_torque_scale = 2.0; // 피드백 토크 스케일링 추가
    bool m_auto_centering = true; // 자동 중심 잡기 활성화

    // smoothing
    double m_smoothing_factor = 0.05;
    double m_current_steering_angle = 0.0;

    // variables
    ros_g29_force_feedback::ForceFeedback m_target;
    bool m_is_target_updated = false;
    bool m_is_brake_range = false;
    struct ff_effect m_effect;
    double m_position;
    double m_torque;
    double m_attack_length;

public:
    G29ForceFeedback();
    ~G29ForceFeedback();

private:
    void targetCallback(const ros_g29_force_feedback::ForceFeedback &in_target);
    void carlaSteeringCallback(const std_msgs::Float64 &steering_msg);
    void loop(const ros::TimerEvent&);
    int testBit(int bit, unsigned char *array);
    void initDevice();
    std::string findDeviceByName(const std::string& device_name);
    void calcRotateForce(double &torque, double &attack_length, const ros_g29_force_feedback::ForceFeedback &target, const double &current_position);
    void calcCenteringForce(double &torque, const ros_g29_force_feedback::ForceFeedback &target, const double &current_position);
    void uploadForce(const double &position, const double &force, const double &attack_length);
};

G29ForceFeedback::G29ForceFeedback() {
    ros::NodeHandle n;
    sub_target = n.subscribe("/ff_target", 1, &G29ForceFeedback::targetCallback, this);
    sub_carla_steering = n.subscribe("/carla/steering_angle", 1, &G29ForceFeedback::carlaSteeringCallback, this);

    n.getParam("loop_rate", m_loop_rate);
    n.getParam("max_torque", m_max_torque);
    n.getParam("min_torque", m_min_torque);
    n.getParam("brake_position", m_brake_position);
    n.getParam("brake_torque_rate", m_brake_torque_rate);
    n.getParam("auto_centering_max_torque", m_auto_centering_max_torque);
    n.getParam("auto_centering_max_position", m_auto_centering_max_position);
    n.getParam("eps", m_eps);
    n.getParam("auto_centering", m_auto_centering);
    n.getParam("smoothing_factor", m_smoothing_factor);
    n.getParam("min_feedback_torque", m_min_feedback_torque);
    n.getParam("feedback_torque_scale", m_feedback_torque_scale);

    m_device_name = findDeviceByName("Logitech G29 Driving Force Racing Wheel");

    initDevice();

    ros::Duration(1).sleep();
    timer = n.createTimer(ros::Duration(m_loop_rate), &G29ForceFeedback::loop, this);
}

G29ForceFeedback::~G29ForceFeedback() {
    m_effect.type = FF_CONSTANT;
    m_effect.id = -1;
    m_effect.u.constant.level = 0;
    m_effect.direction = 0;
    if (ioctl(m_device_handle, EVIOCSFF, &m_effect) < 0) {
        std::cout << "failed to upload m_effect" << std::endl;
    }
}

std::string G29ForceFeedback::findDeviceByName(const std::string& device_name) {
    const char *dir_path = "/dev/input/";
    struct dirent *entry;
    DIR *dp = opendir(dir_path);

    if (dp == nullptr) {
        ROS_ERROR("Cannot open input directory");
        return "";
    }

    while ((entry = readdir(dp))) {
        if (strncmp(entry->d_name, "event", 5) == 0) {
            std::string dev_path = std::string(dir_path) + entry->d_name;
            int fd = open(dev_path.c_str(), O_RDONLY);
            if (fd < 0) continue;

            char name[256] = "Unknown";
            ioctl(fd, EVIOCGNAME(sizeof(name)), name);
            close(fd);

            if (device_name == name) {
                closedir(dp);
                return dev_path;
            }
        }
    }

    closedir(dp);
    ROS_ERROR("Device not found: %s", device_name.c_str());
    return "";
}

void G29ForceFeedback::loop(const ros::TimerEvent&) {
    struct input_event event;
    double last_position = m_position;
    while (read(m_device_handle, &event, sizeof(event)) == sizeof(event)) {
        if (event.type == EV_ABS && event.code == m_axis_code) {
            m_position = (event.value - (m_axis_max + m_axis_min) * 0.5) * 2 / (m_axis_max - m_axis_min);
        }
    }

    if (m_is_brake_range || m_auto_centering) {
        calcCenteringForce(m_torque, m_target, m_position);
        m_attack_length = 0.0;
    } else {
        calcRotateForce(m_torque, m_attack_length, m_target, m_position);
        m_is_target_updated = false;
    }

    uploadForce(m_target.position, m_torque, m_attack_length);
}

void G29ForceFeedback::calcRotateForce(double &torque,
                                       double &attack_length,
                                       const ros_g29_force_feedback::ForceFeedback &target,
                                       const double &current_position) {
    double diff = target.position - current_position;
    double direction = (diff > 0.0) ? 1.0 : -1.0;

    if (fabs(diff) < m_eps) {
        torque = 0.0;
        attack_length = 0.0;
    } else if (fabs(diff) < m_brake_position) {
        m_is_brake_range = true;
        torque = target.torque * m_brake_torque_rate * -direction;
        attack_length = m_loop_rate;
    } else {
        torque = target.torque * direction;
        attack_length = m_loop_rate;
    }

    // 최소 피드백 토크 적용
    if (fabs(torque) < m_min_feedback_torque) {
        torque = m_min_feedback_torque * direction;
    }

    // 피드백 토크 스케일링 적용
    torque *= m_feedback_torque_scale;
}

void G29ForceFeedback::calcCenteringForce(double &torque,
                                          const ros_g29_force_feedback::ForceFeedback &target,
                                          const double &current_position) {
    double diff = target.position - current_position;
    double direction = (diff > 0.0) ? 1.0 : -1.0;

    if (fabs(diff) < m_eps) {
        torque = 0.0;
    } else {
        double torque_range = m_auto_centering_max_torque - m_min_torque;
        double power = (fabs(diff) - m_eps) / (m_auto_centering_max_position - m_eps);
        double buf_torque = power * torque_range + m_min_torque;
        torque = std::min(buf_torque, m_auto_centering_max_torque) * direction;

        // 최소 피드백 토크 적용
        if (fabs(torque) < m_min_feedback_torque) {
            torque = m_min_feedback_torque * direction;
        }

        // 피드백 토크 스케일링 적용
        torque *= m_feedback_torque_scale;
    }
}

void G29ForceFeedback::uploadForce(const double &position,
                                   const double &torque,
                                   const double &attack_length) {
    m_effect.u.constant.level = 0x7fff * std::min(torque, m_max_torque);
    m_effect.direction = 0xC000;
    m_effect.u.constant.envelope.attack_level = 0;
    m_effect.u.constant.envelope.attack_length = attack_length;
    m_effect.u.constant.envelope.fade_level = 0;
    m_effect.u.constant.envelope.fade_length = attack_length;

    if (ioctl(m_device_handle, EVIOCSFF, &m_effect) < 0) {
        std::cout << "failed to upload effect" << std::endl;
    }
}

void G29ForceFeedback::targetCallback(const ros_g29_force_feedback::ForceFeedback &in_msg) {
    if (m_target.position == in_msg.position && m_target.torque == fabs(in_msg.torque)) {
        m_is_target_updated = false;
    } else {
        m_target = in_msg;
        m_target.torque = fabs(m_target.torque);
        m_is_target_updated = true;
        m_is_brake_range = false;
    }
}

void G29ForceFeedback::carlaSteeringCallback(const std_msgs::Float64 &steering_msg) {
    double target_steering = steering_msg.data;
    
    // 스무딩 적용
    m_current_steering_angle = (1.0 - m_smoothing_factor) * m_current_steering_angle + m_smoothing_factor * target_steering;

    ros_g29_force_feedback::ForceFeedback target;
    target.position = m_current_steering_angle;
    target.torque = m_max_torque;  // 필요에 따라 이 값을 조정하세요.
    targetCallback(target);
}

void G29ForceFeedback::initDevice() {
    unsigned char key_bits[1+KEY_MAX/8/sizeof(unsigned char)];
    unsigned char abs_bits[1+ABS_MAX/8/sizeof(unsigned char)];
    unsigned char ff_bits[1+FF_MAX/8/sizeof(unsigned char)];
    struct input_event event;
    struct input_absinfo abs_info;

    m_device_handle = open(m_device_name.c_str(), O_RDWR|O_NONBLOCK);
    if (m_device_handle == -1) {
        ROS_ERROR("Failed to open device: %s", m_device_name.c_str());
        return;
    } else {
        ROS_INFO("Successfully opened device: %s", m_device_name.c_str());
    }

    struct input_absinfo absinfo;
    if (ioctl(m_device_handle, EVIOCGABS(m_axis_code), &absinfo) == -1) {
        ROS_ERROR("Cannot get axis range for axis code: %d", m_axis_code);
        return;
    } else {
        m_axis_min = absinfo.minimum;
        m_axis_max = absinfo.maximum;
        ROS_INFO("Axis range retrieved: min=%d, max=%d", m_axis_min, m_axis_max);
    }
    if (m_device_handle < 0) {
        std::cout << "ERROR: cannot open device : "<< m_device_name << std::endl;
        exit(1);
    } else {
        std::cout << "device opened" << std::endl;
    }

    memset(abs_bits, 0, sizeof(abs_bits));
    if (ioctl(m_device_handle, EVIOCGBIT(EV_ABS, sizeof(abs_bits)), abs_bits) < 0) {
        std::cout << "ERROR: cannot get abs bits" << std::endl;
        exit(1);
    }

    memset(ff_bits, 0, sizeof(ff_bits));
    if (ioctl(m_device_handle, EVIOCGBIT(EV_FF, sizeof(ff_bits)), ff_bits) < 0) {
        std::cout << "ERROR: cannot get ff bits" << std::endl;
        exit(1);
    }

    if (ioctl(m_device_handle, EVIOCGABS(m_axis_code), &abs_info) < 0) {
        std::cout << "ERROR: cannot get axis range" << std::endl;
        exit(1);
    }
    m_axis_max = abs_info.maximum;
    m_axis_min = abs_info.minimum;
    if (m_axis_min >= m_axis_max) {
        std::cout << "ERROR: axis range has bad value" << std::endl;
        exit(1);
    }

    if (!testBit(FF_CONSTANT, ff_bits)) {
        std::cout << "ERROR: force feedback is not supported" << std::endl;
        exit(1);
    } else {
        std::cout << "force feedback supported" << std::endl;
    }

    memset(&event, 0, sizeof(event));
    event.type = EV_FF;
    event.code = FF_AUTOCENTER;
    event.value = 0;
    if (write(m_device_handle, &event, sizeof(event)) != sizeof(event)) {
        std::cout << "failed to disable auto centering" << std::endl;
        exit(1);
    }

    memset(&m_effect, 0, sizeof(m_effect));
    m_effect.type = FF_CONSTANT;
    m_effect.id = -1;
    m_effect.trigger.button = 0;
    m_effect.trigger.interval = 0;
    m_effect.replay.length = 0xffff;
    m_effect.replay.delay = 0;
    m_effect.u.constant.level = 0;
    m_effect.direction = 0xC000;
    m_effect.u.constant.envelope.attack_length = 0;
    m_effect.u.constant.envelope.attack_level = 0;
    m_effect.u.constant.envelope.fade_length = 0;
    m_effect.u.constant.envelope.fade_level = 0;

    if (ioctl(m_device_handle, EVIOCSFF, &m_effect) < 0) {
        std::cout << "failed to upload m_effect" << std::endl;
        exit(1);
    }

    memset(&event, 0, sizeof(event));
    event.type = EV_FF;
    event.code = m_effect.id;
    event.value = 1;
    if (write(m_device_handle, &event, sizeof(event)) != sizeof(event)) {
        std::cout << "failed to start event" << std::endl;
        exit(1);
    }
}

int G29ForceFeedback::testBit(int bit, unsigned char *array) {
    return ((array[bit / (sizeof(unsigned char) * 8)] >> (bit % (sizeof(unsigned char) * 8))) & 1);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_g29_force_feedback_node");
    G29ForceFeedback g29_ff;
    ros::spin();
    return 0;
}
