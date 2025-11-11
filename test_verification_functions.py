import time
import numpy as np
from robot_api import robot
from sim import pnp

"""
Example script for how to use the verification functions in robot_api
This script tests afunctions: is_gripper_open(), is_robot_conf_at(), is_robot_ee_at(), is_obj_at()
Run this script to check how to use the functions.
"""


def test_is_gripper_open():
    """
    Test the gripper state verification function
    
    """
    print("\n")
    print("TEST 1: is_gripper_open()\n")

    #Test1 :We open the gripper
    print("\nOpening gripper")
    robot.release()

    result = robot.is_gripper_open()
    print(f"Result: {result}")
    if result == True :
        print(f"SUCCESS: Gripper detected as open" )
    else:
        print(f"FAIL: Expected True, we had {result}")

    # Test 2: Close the gripper
    print("\nClosing gripper")
    robot.grasp()

    result = robot.is_gripper_open()
    print(f"Result: {result}")
    if result == False :
        print(f"SUCESS Gripper detected as closed") 
    else:
        print(f"FAIL: Expected False, we had {result}")


def test_is_robot_conf_at():
    """
    
    Tests the robot configuration verification function
    """
    print("\n")
    print("TEST 2: is_robot_conf_at()")
    

    # Initialize tHE robor to home position
    print("\nTest 2 : Moving to home position")
    robot.initialize()

    #we test if robot is at home
    
    home_qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]

    result = robot.is_robot_conf_at(home_qpos)
    print(f"Expected config: {home_qpos}")
    print(f"Result: {result}")
    if result == True:
        print(f"SUCESS Robot at home configuration" )
    else:
         print(f"FAIL: Expected True, we got {result}")

    # Test with wrong configuration
    print("\nTest with wrong configuration:")
    wrong_config = [0, 0, 0, 0, 0, 0]
    result = robot.is_robot_conf_at(wrong_config)
    if result == False:
        print(f"SUCESS Correctly detected wrong configuration") 
    else:
        print(f"FAIL: Expected False, we got {result}")


def test_is_gripper_tip_at():
    """Test the gripper  tip position """
    print("\n")
    print("TEST 3: is_gripper_tip_at()")

    # Move to a position we know
    print("\nTest 3:  Moving to position [0.3, 0.3, 0.6]")
    target_pos = np.array([0.3, 0.3, 0.6])

    # Get current configuration and plan trajectory
    current_q = pnp.get_joints()
    target_q, traj = pnp.plan_trajectory_from_config(current_q, target_pos, 2.0)

    if target_q is None:
        print(f"FAIL: Could not plan trajectory to position")
        return
    # Execute trajectory manually
    while len(traj) > 0:
        q = traj.popleft()
        pnp.set_actuators(q)
        robot.step_simulation()
        time.sleep(robot.model.opt.timestep)

    result = robot.is_gripper_tip_at(target_pos)
    print(f"Result: {result}")
    if result == True:
        print(f"SUCCESS: Gripper tip at target position" )
    else:
        print(f"FAIL: Expected True, we got {result}")

    # Test with wrong position
    print("\nTest 3:Testing with wrong position")
    wrong_pos = [0.5, 0.5, 0.8]
    result = robot.is_gripper_tip_at(wrong_pos)
    print(f"Testing position: {wrong_pos}")
    print(f"Result: {result}")
    if result == False:
        print(f"SUCCESS: Correctly detected wrong position")
    else:
        print(f"FAIL: Expected False, got {result}")


def test_is_obj_at():
    """
    Test the object position verification
    """
    print("\n" )
    print("TEST 4: is_obj_at()")

    #red box position
    print("\nTest4: Checking red_box at initial position")

    #THe currentposition
    actual_pos = pnp.get_body_pos("red_box")
    print(f"Red box actual position: {actual_pos}")

    #CHeck if object is at its position
    result = robot.is_obj_at("red_box", actual_pos)
    print(f"Result: {result}")
    if result == True:
        print(f"SUCESS Object at expected position")
    else:
        print(f"FAIL: Expected True, we got {result}")

    #Test with wrong position
    print("\nTest 4: Testing with wrong position")
    wrong_pos = [1.0, 1.0, 1.0]
    result = robot.is_obj_at("red_box", wrong_pos)
    print(f"Testing position: {wrong_pos}")
    print(f"Result: {result}")
    if result == False:
        print(f"SUCESS Correctly detected wrong position")
    else:
        f"FAIL: Expected False, we got {result}"


def main():

    print("\n")
    #We launch the viewar
    robot.launch_viewer()
    time.sleep(1)

    #And init the robot
    robot.initialize()

    #Run all thetests
    try:
        print("-" *30)
        test_is_gripper_open()
        print("-" *30)
        test_is_robot_conf_at()
        print("-" *30)
        test_is_gripper_tip_at()
        print("-" *30)
        test_is_obj_at()

        print("All tests completed")
        # Keep simulation running
        while robot.viewer and robot.viewer.is_running():
            robot.step_simulation()
            time.sleep(robot.model.opt.timestep)

    except Exception as e:
        print(f"\n\nERROR: {e}")

    finally:
        print("\nClosing viewer")
        robot.close_viewer()


if __name__ == "__main__":
    main()
