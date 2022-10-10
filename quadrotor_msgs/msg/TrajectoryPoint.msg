#reused from uzh_quadrotor_commons
duration time_from_start

geometry_msgs/Vector3 position

geometry_msgs/Vector3 velocity

geometry_msgs/Vector3 acceleration

geometry_msgs/Vector3 jerk

geometry_msgs/Vector3 snap

# Heading angle with respect to world frame [rad]
float64 heading

# First derivative of the heading angle [rad/s]
float64 heading_rate

# Second derivative of the heading angle [rad/s^2]
float64 heading_acceleration