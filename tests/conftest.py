import os
import sys
import json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define the ML Test Score sections based on test file names
ML_TEST_SECTIONS = {
    "test_feature_and_data.py": "Data Tests",
    "test_ml_infrastructure.py": "ML Infrastructure Tests",
    "test_model_development.py": "Model Development Tests",
    "test_monitoring.py": "Monitoring Tests",
    "test_mutamorphic.py": "Mutamorphic Testing",
}

# Store test outcomes here
test_results = defaultdict(list)

def get_score_interpretation(score):
    if score == 0:
        return "More of a research project than a productionized system."
    elif 0 < score <= 1:
        return "Not totally untested, but it is worth considering the possibility of serious holes in reliability."
    elif 1 < score <= 2:
        return "There’s been a first pass at basic productionization, but additional investment may be needed."
    elif 2 < score <= 3:
        return "Reasonably tested, but it’s possible that more of those tests and procedures may be automated."
    elif 3 < score <= 5:
        return "Strong levels of automated testing and monitoring, appropriate for mission-critical systems."
    else:  # score > 5
        return "Exceptional levels of automated testing and monitoring."

def save_ml_test_score(section_scores, final_score):
    results = {
        "sections": section_scores,
        "final_score": final_score
    }
    scores_path = Path(__file__).parent.parent / "metrics" / "ml_scores.json"

    if not scores_path.parent.exists():
        scores_path.parent.mkdir(parents=True)

    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)

def generate_badge(score):
    try:
        import anybadge

        thresholds = {
            1: 'red',         
            2: 'orange',      
            3: 'yellow',      
            5: 'yellowgreen', 
            6: 'green'        
        }

        badge = anybadge.Badge(
            label='ML Test Score',
            value=f'{score:.1f}',
            thresholds=thresholds,
            default_color='red'
        )

        badge_path = Path(__file__).parent.parent / "metrics" / "ml_test_score.svg"
        badge.write_badge(badge_path, overwrite=True)
        print(f"ML Test Score badge saved to {badge_path}")
    except ImportError:
        print("Warning: anybadge not installed. Badge not generated.")
    except Exception as e:
        print(f"Error generating badge: {str(e)}")


# Hook to collect test results
def pytest_runtest_logreport(report):
    if report.when == "call":
        test_file = os.path.basename(report.fspath)
        section = ML_TEST_SECTIONS.get(test_file, "Uncategorized")
        test_results[section].append(report.outcome)

# Hook to compute score at the end
def pytest_sessionfinish(session, exitstatus):
    section_scores = {}
    for section, outcomes in test_results.items():
        passed_tests = outcomes.count("passed")
        total_tests = len(outcomes)
        section_scores[section] = {
            "score": passed_tests,
            "max_score": total_tests
        }

    if section_scores:
        min_score = min(score["score"] for score in section_scores.values())
        final_score = min_score
    else:
        final_score = 0

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_sep("=", "ML Test Score", bold=True)
        for section, score_data in section_scores.items():
            reporter.write_line(f"{section}: {score_data['score']}/{score_data['max_score']}")
        reporter.write_line("")
        reporter.write_line(f"Final ML Test Score: {final_score}")
        reporter.write_line(f"Interpretation: {get_score_interpretation(final_score)}")

    save_ml_test_score(section_scores, final_score)
    generate_badge(final_score)
