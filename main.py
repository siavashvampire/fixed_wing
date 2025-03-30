from pyfly import PyFly

if __name__ == "__main__":
    from dryden import DrydenGustModel
    from pid_controller import PIDController

    pfly = PyFly("pyfly_config.json", "x8_param.mat")
    pfly.seed(0)

    pid = PIDController(pfly.dt)
    pid.set_reference(phi=0.2, theta=0, va=22)

    pfly.reset(state={"roll": -0.5, "pitch": 0.15})

    for i in range(1000):
        phi = pfly.state["roll"].value
        theta = pfly.state["pitch"].value
        Va = pfly.state["Va"].value
        omega = [pfly.state["omega_p"].value, pfly.state["omega_q"].value, pfly.state["omega_r"].value]

        action = pid.get_action(phi, theta, Va, omega)
        success, step_info = pfly.step(action)

        if not success:
            break

    pfly.render(block=True)