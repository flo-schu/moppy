import os
import click

from pymob.sim.config import Config
from pymob.utils import help

@click.command()
@click.option("-c", "--case_study", type=str, default="lotka_volterra_case_study", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="test_scenario", 
              help=help.scenario)
@click.option("-p", "--package", type=str, default="case_studies", 
              help=help.package)
@click.option("-l", "--logging", type=str, default=None)
@click.option("-f", "--logfile", type=str, default=None)
def main(case_study, scenario, package, logging, logfile):

    cfg = os.path.join(package, case_study, "scenarios", scenario, "settings.cfg")
    config = Config(cfg)
    config.case_study.name = case_study
    config.case_study.scenario = scenario
    config.import_casestudy_modules(reset_path=True)

    # update parameters from config file if they are specified
    if logging is not None: config.case_study.logging = logging
    if logfile is not None: config.case_study.logfile = logfile

    # import simulation      
    Simulation = config.import_simulation_from_case_study()
    sim = Simulation(config)
    sim.setup()

    # run the simulation
    evaluator = sim.dispatch(theta=sim.model_parameter_dict)
    evaluator()

    # store and process output
    sim.dump(results=evaluator.results)
    sim.plot(results=evaluator.results)

if __name__ == "__main__":
    main()