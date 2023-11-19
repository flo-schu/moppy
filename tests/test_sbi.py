import socket
import os
import pytest


class TestSbiPipeline:
    """tests sbi pipeline"""
    case_study = "core_daphnia"
    scenario = "test"

    def test_generate_sims(self):
        pytest.skip()   
        from moppy import generate_sims
        print(f"hostname: {socket.gethostname()}, user: {os.environ.get('USERNAME')}")
        generate_sims.main(
            case_study=[self.case_study, self.scenario],
            n_sims=5,
            worker="1"
        )
        generate_sims.main(
            case_study=[self.case_study, self.scenario],
            n_sims=5,
            worker="2"
        )

    def test_process_simulations(self):
        pytest.skip()   
        from moppy import process_simulations
        print(f"hostname: {socket.gethostname()}, user: {os.environ.get('USERNAME')}")
        process_simulations.main(
            case_study=[self.case_study, self.scenario],
        )

    def test_prior_predictive_checks(self):
        pytest.skip()   
        from moppy import prior_predictive_checks
        prior_predictive_checks.main(
            case_study=[self.case_study, self.scenario]
        )

    def test_train_network_SNPE(self):
        pytest.skip()   
        from moppy import train_network
        print(f"hostname: {socket.gethostname()}, user: {os.environ.get('USERNAME')}")
        train_network.main(
            case_study=[self.case_study, self.scenario],
            inferer_engine="SNPE", 
            training_batch_size=1
        )

    def test_train_network_SNLE(self):
        pytest.skip()   
        from moppy import train_network
        print(f"hostname: {socket.gethostname()}, user: {os.environ.get('USERNAME')}")
        train_network.main(
            case_study=[self.case_study, self.scenario],
            inferer_engine="SNLE", 
            training_batch_size=1
        )

    def test_snle_sampling(self):
        pytest.skip()   
        from moppy.inference.sbi import sbi_snle_sample_posterior
        print(f"hostname: {socket.gethostname()}, user: {os.environ.get('USERNAME')}")
        sbi_snle_sample_posterior.main(
            case_study=[self.case_study, self.scenario],
        )


    def test_evaluate_sbi(self):
        pytest.skip()   
        from moppy import evaluate_sbi
        evaluate_sbi.main(
            case_study=[self.case_study, self.scenario],
        )

    def test_posterior_predictions(self):
        pytest.skip()   
        from moppy import posterior_predictions
        posterior_predictions.main(
            case_study=[self.case_study, self.scenario],
            posterior_cluster=0,
            nsims=2
        )

    def test_plot_posterior_predictions(self):
        pytest.skip()   
        from moppy import plot_posterior_predictions
        plot_posterior_predictions.main(
            case_study=[self.case_study, self.scenario],
            posterior_cluster=0
        )