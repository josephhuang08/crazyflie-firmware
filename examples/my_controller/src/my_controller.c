#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "app.h"
#include "log.h"
#include "math3d.h"
#include "stabilizer_types.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "NN"
#include "debug.h"

void appMain() {
    DEBUG_PRINT("Waiting for activation ...\n");

    while(1) {
        vTaskDelay(M2T(2000));
        //DEBUG_PRINT("Neural Network Controller Active\n");
    }
}

// The new controller goes here --------------------------------------------
#include "controller.h"
#include "controller_pid.h"

#include "model.h"

// 2D array to match the onnx2c generated input/output
static float nn_obs_space[1][7]; 
static float nn_actions[1][4];

typedef struct {
    float thrust_to_weight;
    float robot_weight;
    float moment_scale;
} ControllerConfig;

void prePhysicsStep(const float actions[4], float* thrust, float moment[3], const ControllerConfig* cfg) {
    // Clamp actions to [-1.0, 1.0]
    float clamped_actions[4];
    for (int i = 0; i < 4; ++i) {
        clamped_actions[i] = fmaxf(-1.0f, fminf(1.0f, actions[i]));
    }

    // Compute thrust
    *thrust = cfg->thrust_to_weight * cfg->robot_weight * (clamped_actions[0] + 1.0f) / 2.0f;

    // Compute moments
    for (int i = 0; i < 3; ++i) {
        moment[i] = cfg->moment_scale * clamped_actions[i + 1];
    }
}

void controllerOutOfTreeInit() {
    // Initialize neural network output
    nn_actions[0][0] = 0.0f;
    nn_actions[0][1] = 0.0f;
    nn_actions[0][2] = 0.0f;
    nn_actions[0][3] = 0.0f;
    DEBUG_PRINT("Neural Network Controller Initialized\n");
}

bool controllerOutOfTreeTest() {
    // Add any test logic here if needed
    return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const stabilizerStep_t stabilizerStep) {
    // Set control mode to force/torque
    control->controlMode = controlModeForceTorque;
    
    // Run at specified frequency
    if (!RATE_DO_EXECUTE(RATE_250_HZ, stabilizerStep)) {
        return;
    }
    
    TickType_t start = xTaskGetTickCount();
    // Build state vector: [ang_vel_x, ang_vel_y, ang_vel_z, , quat_w, quat_x, quat_y, quat_z]
    nn_obs_space[0][0] = radians(sensors->gyro.x);
    nn_obs_space[0][1] = radians(sensors->gyro.y);
    nn_obs_space[0][2] = radians(sensors->gyro.z);
    nn_obs_space[0][3] = state->attitudeQuaternion.w;
    nn_obs_space[0][4] = state->attitudeQuaternion.x;
    nn_obs_space[0][5] = state->attitudeQuaternion.y;
    nn_obs_space[0][6] = state->attitudeQuaternion.z;
    
    // Inference RL policy
    networkInference(nn_obs_space, nn_actions);

    // Process NN output through prePhysicsStep before assigning to control
    static ControllerConfig cfg = {.thrust_to_weight = 1.9f, .robot_weight = 0.3f, .moment_scale = 0.01f };
    float thrust;
    float moment[3];
    prePhysicsStep(nn_actions[0], &thrust, moment, &cfg);
    control->thrustSi = thrust;
    control->torqueX = moment[0];
    control->torqueY = moment[1];
    control->torqueZ = moment[2];

    TickType_t end = xTaskGetTickCount();

    if (stabilizerStep % 1000 == 0) { 
        DEBUG_PRINT("setpoint->mode.x: %d\n", setpoint->mode.x);
        DEBUG_PRINT("setpoint->mode.y: %d\n", setpoint->mode.y);
        DEBUG_PRINT("setpoint->mode.z: %d\n", setpoint->mode.z);
        DEBUG_PRINT("setpoint->mode.roll: %d\n", setpoint->mode.roll);
        DEBUG_PRINT("setpoint->mode.pitch: %d\n", setpoint->mode.pitch);
        DEBUG_PRINT("setpoint->mode.yaw: %d\n", setpoint->mode.yaw);
        DEBUG_PRINT("setpoint->mode.quat: %d\n", setpoint->mode.quat);

        DEBUG_PRINT("Exec time: %lu ms\n", (unsigned long)((end - start) * portTICK_PERIOD_MS));
        DEBUG_PRINT("NN output: thrust=%.3f, tx=%.3f, ty=%.3f, tz=%.3f\n",
        (double)nn_actions[0][0], (double)nn_actions[0][1], (double)nn_actions[0][2], (double)nn_actions[0][3]);
        DEBUG_PRINT("Control: thrust=%.3f, tx=%.3f, ty=%.3f, tz=%.3f\n\n",
        (double)control->thrustSi, (double)control->torqueX, (double)control->torqueY, (double)control->torqueZ);
    }
}


// Logging

/**
 * Log variables of OOT controller
 */
LOG_GROUP_START(controllerOOT)
/**
 * @brief NN Input: angular velocity x (rad/s)
 */
LOG_ADD(LOG_FLOAT, ang_vel_x, &nn_obs_space[0][0])
/**
 * @brief NN Input: angular velocity y (rad/s)
 */
LOG_ADD(LOG_FLOAT, ang_vel_y, &nn_obs_space[0][1])
/**
 * @brief NN Input: angular velocity z (rad/s)
 */
LOG_ADD(LOG_FLOAT, ang_vel_z, &nn_obs_space[0][2])
/**
 * @brief NN output before pre-calculation: thrust
 */
LOG_ADD(LOG_FLOAT, thrust, &nn_actions[0][0])
/**
 * @brief NN output before pre-calculation: torqueX
 */
LOG_ADD(LOG_FLOAT, tx, &nn_actions[0][1])
/**
 * @brief NN output before pre-calculation: torqueY
 */
LOG_ADD(LOG_FLOAT, ty, &nn_actions[0][2])
/**
 * @brief NN output before pre-calculation: torqueZ
 */
LOG_ADD(LOG_FLOAT, tz, &nn_actions[0][3])
LOG_GROUP_STOP(controllerOOT)