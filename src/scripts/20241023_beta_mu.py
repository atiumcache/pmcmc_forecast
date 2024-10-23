from src.pmcmc.main import main as pmcmc_main
import sys


if __name__ == "__main__":
    # only command line arg is the target_date
    date = sys.argv[1]
    pmcmc_main(location_code="06", target_date=date)
