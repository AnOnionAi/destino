import nevergrad
import ray
import muzero
import os
import torch


class Hyperparams:
    def __init__(self):
        self.budget = 20
        self.parallel_experiments = 1
        self.lr = nevergrad.p.Log(lower=0.0001, upper=1)
        self.discount = nevergrad.p.Log(lower=0.95, upper=1)
        self.parametrization = nevergrad.p.Dict(lr=self.lr, discount=self.discount)

    def hyperparameter_search(self, game_name, num_tests=20):
        """
        Search for hyperparameters by launching parallel experiments.

        Args:
            game_name (str): Name of the game module, it should match the name of a .py file
            in the "./games" directory.

            parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

            budget (int): Number of experiments to launch in total.

            parallel_experiments (int): Number of experiments to launch in parallel.

            num_tests (int): Number of games to average for evaluating an experiment.
        """
        print("Budget", self.budget)
        optimizer = nevergrad.optimizers.OnePlusOne(
            parametrization=self.parametrization, budget=self.budget
        )

        running_experiments = []
        best_training = None
        try:
            # Launch initial experiments
            for _ in range(self.parallel_experiments):
                if self.budget > 0:
                    param = optimizer.ask()
                    print(f"Launching new experiment: {param.value}")
                    muz = muzero.MuZero(game_name, param.value)
                    muz.param = param
                    # muz.gpu_config()
                    # muz.cpu_actoring()
                    muz.train(False)
                    running_experiments.append(muz)
                    self.budget -= 1

            while self.budget > 0 or any(running_experiments):
                for i, experiment in enumerate(running_experiments):
                    if experiment and experiment.config.training_steps <= ray.get(
                        experiment.shared_storage_worker.get_info.remote(
                            "training_step"
                        )
                    ):
                        experiment.terminate_workers()
                        result = experiment.test(False, num_tests=num_tests)
                        if not best_training or best_training["result"] < result:
                            best_training = {
                                "result": result,
                                "config": experiment.config,
                                "checkpoint": experiment.checkpoint,
                            }
                        print(f"Parameters: {experiment.param.value}")
                        print(f"Result: {result}")
                        optimizer.tell(experiment.param, -result)

                        if self.budget > 0:
                            param = optimizer.ask()
                            print(f"Launching new experiment: {param.value}")
                            muz = muzero.MuZero(game_name, param.value)
                            muz.param = param
                            # muz.gpu_config()
                            # muz.cpu_actoring()
                            muz.train(False)
                            running_experiments[i] = muz
                            self.budget -= 1
                        else:
                            running_experiments[i] = None

        except KeyboardInterrupt:
            for experiment in running_experiments:
                if isinstance(experiment, muz):
                    experiment.terminate_workers()

        recommendation = optimizer.provide_recommendation()
        print("Best hyperparameters:")
        print(recommendation.value)
        if best_training:
            # Save best training weights (but it's not the recommended weights)
            # This might need to be updated because results_path is now on the muzero object
            os.makedirs(best_training["config"].results_path, exist_ok=True)
            torch.save(
                best_training["checkpoint"],
                os.path.join(best_training["config"].results_path, "model.checkpoint"),
            )
            with open(
                os.path.join(
                    best_training["config"].results_path, "best_parameters.txt"
                ),
                "w",
            ) as text_file:
                text_file.write(str(recommendation.value))
        return recommendation.value
