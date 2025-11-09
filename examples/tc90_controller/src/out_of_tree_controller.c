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
static float nn_obs_space[1][8]; 
static float nn_actions[1][3];

typedef struct {
    float thrust_to_weight;
    float robot_weight;
    float moment_scale;
} ControllerConfig;

void prePhysicsStep(const float actions[3], float moment[3], const ControllerConfig* cfg) {
    // Clamp actions to [-1.0, 1.0]
    float clamped_actions[3];
    for (int i = 0; i < 3; ++i) {
        clamped_actions[i] = fmaxf(-1.0f, fminf(1.0f, actions[i]));
    }

    // Compute moments (NN outputs torques directly)
    for (int i = 0; i < 3; ++i) {
        moment[i] = cfg->moment_scale * clamped_actions[i];
    }
}

void controllerOutOfTreeInit() {
    // Initialize neural network output
    nn_actions[0][0] = 0.0f;
    nn_actions[0][1] = 0.0f;
    nn_actions[0][2] = 0.0f;
    DEBUG_PRINT("Neural Network Controller Initialized\n");
}

bool controllerOutOfTreeTest() {
    // Add any test logic here if needed
    return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const stabilizerStep_t stabilizerStep) {
    // Set control mode to force/torque
    control->controlMode = controlModeForceTorque;
    
    // // Run at specified frequency
    if (!RATE_DO_EXECUTE(RATE_500_HZ, stabilizerStep)) {
        return;
    }
    
    //TickType_t start = xTaskGetTickCount();
    // Build state vector: [ang_vel_x, ang_vel_y, ang_vel_z, quat_w, quat_x, quat_y, quat_z, thrust_setpoint_N]
    nn_obs_space[0][0] = radians(sensors->gyro.x);
    nn_obs_space[0][1] = radians(sensors->gyro.y);
    nn_obs_space[0][2] = radians(sensors->gyro.z);
    nn_obs_space[0][3] = state->attitudeQuaternion.w;
    nn_obs_space[0][4] = state->attitudeQuaternion.x;
    nn_obs_space[0][5] = state->attitudeQuaternion.y;
    nn_obs_space[0][6] = state->attitudeQuaternion.z;
    // Convert Xbox controller thrust from uint16_t [0, 65535] to Newtons [0, 0.45]
    // Using THRUST_MAX = 0.1125f per motor, 4 motors total = 0.45f max thrust
    nn_obs_space[0][7] = (setpoint->thrust / 65535.0f) * 0.45f;
    
    // Inference RL policy
    networkInference(nn_obs_space, nn_actions);

    // Thrust comes directly from Xbox controller (already converted to Newtons)
    control->thrustSi = nn_obs_space[0][7]; // Use the converted thrust value
    
    // Process NN output (torques only) through prePhysicsStep
    static ControllerConfig cfg = {.thrust_to_weight = 1.9f, .robot_weight = 0.3f, .moment_scale = 0.01f };
    float moment[3];
    prePhysicsStep(nn_actions[0], moment, &cfg);
    control->torqueX = moment[0];
    control->torqueY = moment[1];
    control->torqueZ = moment[2];

    //TickType_t end = xTaskGetTickCount();

    // if (stabilizerStep % 1000 == 0) { 
    //     DEBUG_PRINT("Exec time: %lu ms\n", (unsigned long)((end - start) * portTICK_PERIOD_MS));
    //     DEBUG_PRINT("NN output: tx=%.3f, ty=%.3f, tz=%.3f\n",
    //     (double)nn_actions[0][0], (double)nn_actions[0][1], (double)nn_actions[0][2]);
    //     DEBUG_PRINT("Control: thrust=%.3f, tx=%.3f, ty=%.3f, tz=%.3f\n\n",
    //     (double)control->thrustSi, (double)control->torqueX, (double)control->torqueY, (double)control->torqueZ);
    // }
}


// Logging

// /**
//  * Log variables of OOT controller
//  */
// LOG_GROUP_START(controllerOOT)
// /**
//  * @brief NN Input: angular velocity x (rad/s)
//  */
// LOG_ADD(LOG_FLOAT, ang_vel_x, &nn_obs_space[0][0])
// /**
//  * @brief NN Input: angular velocity y (rad/s)
//  */
// LOG_ADD(LOG_FLOAT, ang_vel_y, &nn_obs_space[0][1])
// /**
//  * @brief NN Input: angular velocity z (rad/s)
//  */
// LOG_ADD(LOG_FLOAT, ang_vel_z, &nn_obs_space[0][2])
// /**
//  * @brief NN Input: thrust setpoint from Xbox controller
//  */
// LOG_ADD(LOG_FLOAT, thrust_setpoint, &nn_obs_space[0][7])
// /**
//  * @brief NN output before pre-calculation: torqueX
//  */
// LOG_ADD(LOG_FLOAT, nn_tx, &nn_actions[0][0])
// /**
//  * @brief NN output before pre-calculation: torqueY
//  */
// LOG_ADD(LOG_FLOAT, nn_ty, &nn_actions[0][1])
// /**
//  * @brief NN output before pre-calculation: torqueZ
//  */
// LOG_ADD(LOG_FLOAT, nn_tz, &nn_actions[0][2])
// LOG_GROUP_STOP(controllerOOT)