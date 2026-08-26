"""
Automated Test Suite for Groundwater AI Screening Web Application.
Executes 12 Comprehensive Validation, Domain Applicability, Uncertainty, and Decision Logic Tests.
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from services.model_loader import ModelLoader
from services.prediction_service import PredictionService
from services.uncertainty_service import UncertaintyService
from services.decision_engine import DecisionEngine
from utils.validation import validate_groundwater_sample


def run_tests():
    print("=" * 80)
    print("  GROUNDWATER AI SCREENING APP — 12-CASE SCIENTIFIC VALIDATION TEST SUITE")
    print("=" * 80)

    passed_count = 0
    total_tests = 12

    # TEST 1: Normal In-Domain Sample (pH=7.1, TDS=220, NO3=1.2, Depth=25)
    print("\n[TEST 1] Normal In-Domain Sample (pH=7.1, TDS=220, NO3=1.2, Depth=25)")
    res1 = PredictionService.predict_sample(7.1, 220.0, 1.2, 25.0)
    assert res1["success"] == True and res1["domain_state"] == "IN_DOMAIN", "Test 1 failed!"
    assert res1["results"]["ni"]["decision"]["status_code"] == "BELOW_THRESHOLD", "Test 1 Ni status failed!"
    assert res1["results"]["cd"]["decision"]["status_code"] == "BELOW_THRESHOLD", "Test 1 Cd status failed!"
    from components.report_generator import generate_html_report
    html_rep = generate_html_report(res1["input_parameters"], res1["results"], res1["overall_recommendation"], domain_state=res1["domain_state"])
    assert len(html_rep) > 1000, "HTML Report generation failed!"
    print(f"  🟢 Domain State: {res1['domain_state']}")
    print(f"  ✓ Ni Pred: {res1['results']['ni']['prediction']:.2f} µg/L | Status: {res1['results']['ni']['decision']['title']}")
    print(f"  ✓ Cd Pred: {res1['results']['cd']['prediction']:.2f} µg/L | Status: {res1['results']['cd']['decision']['title']}")
    print(f"  ✓ HTML Report generated successfully ({len(html_rep)} chars)")
    passed_count += 1

    # TEST 2: pH Outside Training Range (pH=6.5, TDS=220, NO3=1.2, Depth=25)
    print("\n[TEST 2] pH Outside Training Range (pH=6.5)")
    res2 = PredictionService.predict_sample(6.5, 220.0, 1.2, 25.0)
    assert res2["success"] == True and res2["domain_state"] == "OUT_OF_DOMAIN", "Test 2 failed!"
    assert res2["results"]["ni"]["decision"]["status_code"] == "OUT_OF_DOMAIN", "Test 2 decision status failed!"
    print(f"  🟡 Domain State: {res2['domain_state']}")
    print(f"  ✓ Warning: {res2['ood_warnings'][0]}")
    passed_count += 1

    # TEST 3: High TDS Out-of-Domain Sample (pH=7.1, TDS=750, NO3=1.2, Depth=25)
    print("\n[TEST 3] High TDS Out-of-Domain Sample (TDS=750)")
    res3 = PredictionService.predict_sample(7.1, 750.0, 1.2, 25.0)
    assert res3["success"] == True and res3["domain_state"] == "OUT_OF_DOMAIN", "Test 3 failed!"
    print(f"  🟡 Domain State: {res3['domain_state']}")
    print(f"  ✓ Warning: {res3['ood_warnings'][0]}")
    passed_count += 1

    # TEST 4: High NO3-N Out-of-Domain Sample (pH=7.1, TDS=220, NO3=15.0, Depth=25)
    print("\n[TEST 4] High NO3-N Out-of-Domain Sample (NO3=15.0)")
    res4 = PredictionService.predict_sample(7.1, 220.0, 15.0, 25.0)
    assert res4["success"] == True and res4["domain_state"] == "OUT_OF_DOMAIN", "Test 4 failed!"
    print(f"  🟡 Domain State: {res4['domain_state']}")
    print(f"  ✓ Warning: {res4['ood_warnings'][0]}")
    passed_count += 1

    # TEST 5: Well Depth Outside Training Range (Depth=80.0)
    print("\n[TEST 5] Well Depth Outside Training Range (Depth=80.0)")
    res5 = PredictionService.predict_sample(7.1, 220.0, 1.2, 80.0)
    assert res5["success"] == True and res5["domain_state"] == "OUT_OF_DOMAIN", "Test 5 failed!"
    print(f"  🟡 Domain State: {res5['domain_state']}")
    print(f"  ✓ Warning: {res5['ood_warnings'][0]}")
    passed_count += 1

    # TEST 6: Negative TDS (TDS = -50)
    print("\n[TEST 6] Negative TDS (TDS = -50)")
    res6 = PredictionService.predict_sample(7.1, -50.0, 1.2, 25.0)
    assert res6["success"] == False and res6["domain_state"] == "INVALID", "Test 6 failed!"
    print(f"  🔴 Domain State: {res6['domain_state']}")
    print(f"  ✓ Rejected: {res6['errors'][0]}")
    passed_count += 1

    # TEST 7: Invalid pH (pH = 20.0)
    print("\n[TEST 7] Invalid pH (pH = 20.0)")
    res7 = PredictionService.predict_sample(20.0, 220.0, 1.2, 25.0)
    assert res7["success"] == False and res7["domain_state"] == "INVALID", "Test 7 failed!"
    print(f"  🔴 Domain State: {res7['domain_state']}")
    print(f"  ✓ Rejected: {res7['errors'][0]}")
    passed_count += 1

    # TEST 8: Missing / Non-Numeric Input
    print("\n[TEST 8] Non-Numeric Input (pH = 'abc')")
    res8 = PredictionService.predict_sample("abc", 220.0, 1.2, 25.0)
    assert res8["success"] == False and res8["domain_state"] == "INVALID", "Test 8 failed!"
    print(f"  🔴 Domain State: {res8['domain_state']}")
    print(f"  ✓ Rejected: {res8['errors'][0]}")
    passed_count += 1

    # TEST 9: Conformal Interval Unavailable Handling
    print("\n[TEST 9] Conformal Interval Unavailable Handling")
    dec_unavail = DecisionEngine.evaluate_metal_screening(
        "ni", 15.0, {"available": False}, domain_state="IN_DOMAIN"
    )
    assert dec_unavail["status_code"] == "CONFIDENCE_UNAVAILABLE", "Test 9 failed!"
    print(f"  🟡 Status: {dec_unavail['title']}")
    passed_count += 1

    # TEST 10: Prediction Interval Overlapping Threshold (Uncertain)
    print("\n[TEST 10] Conformal Interval Overlapping Threshold (Uncertain)")
    unc_result = {"available": True, "lower_bound": 15.0, "upper_bound": 25.0}
    dec_unc = DecisionEngine.evaluate_metal_screening(
        "ni", 18.0, unc_result, domain_state="IN_DOMAIN"
    )
    assert dec_unc["status_code"] == "UNCERTAIN", "Test 10 failed!"
    print(f"  🟡 Status: {dec_unc['title']}")
    passed_count += 1

    # TEST 11: Prediction Interval Strictly Below Threshold
    print("\n[TEST 11] Conformal Interval Strictly Below Threshold")
    below_unc = {"available": True, "lower_bound": 1.0, "upper_bound": 5.0}
    dec_below = DecisionEngine.evaluate_metal_screening(
        "ni", 2.0, below_unc, domain_state="IN_DOMAIN"
    )
    assert dec_below["status_code"] == "BELOW_THRESHOLD", "Test 11 failed!"
    print(f"  🟢 Status: {dec_below['title']}")
    passed_count += 1

    # TEST 12: Potential Threshold Exceedance (Lower Bound > Threshold)
    print("\n[TEST 12] Potential Threshold Exceedance (Lower Bound > Threshold)")
    exceed_unc = {"available": True, "lower_bound": 22.0, "upper_bound": 35.0}
    dec_exceed = DecisionEngine.evaluate_metal_screening(
        "ni", 28.0, exceed_unc, domain_state="IN_DOMAIN"
    )
    assert dec_exceed["status_code"] == "POTENTIAL_EXCEEDANCE", "Test 12 failed!"
    print(f"  🔴 Status: {dec_exceed['title']}")
    passed_count += 1

    print("\n" + "=" * 80)
    print(f"  ALL {passed_count}/{total_tests} SCIENTIFIC VALIDATION & DOMAIN TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
