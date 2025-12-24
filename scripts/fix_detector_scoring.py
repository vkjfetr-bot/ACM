#!/usr/bin/env python3
"""
Replace detector scoring section with helper function call.
"""
import re

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_length = len(content)
    
    # Old scoring section markers
    old_section_start = """        # PERF-03: Only score enabled detectors
        # CRITICAL FIX #6: Replace NaN with 0 after all detector .score() calls to prevent NaN propagation
        # NOTE: Sequential scoring to avoid BLAS/OpenMP thread deadlocks with ThreadPoolExecutor
        with T.section("score.detector_score"):
            Console.info("Starting detector scoring...", component="MODEL")
            score_start_time = time.perf_counter()
            
            # AR1 Detector
            if ar1_enabled and ar1_detector:
                with T.section("score.ar1"):
                    res = ar1_detector.score(score)
                    frame["ar1_raw"] = pd.Series(res, index=frame.index).fillna(0)
                    Console.info(f"AR1 detector scored (samples={len(score)})", component="AR1", samples=len(score))
            
            # PCA Subspace Detector
            if pca_enabled and pca_detector:
                with T.section("score.pca"):
                    pca_spe, pca_t2 = pca_detector.score(score)
                    frame["pca_spe"] = pd.Series(pca_spe, index=frame.index).fillna(0)
                    frame["pca_t2"] = pd.Series(pca_t2, index=frame.index).fillna(0)
                    Console.info(f"PCA detector scored (samples={len(score)})", component="PCA", samples=len(score))
            
            # NOTE: MHAL removed v9.1.0 - redundant with PCA-T2
            
            # Isolation Forest Detector
            if iforest_enabled and iforest_detector:
                with T.section("score.iforest"):
                    res = iforest_detector.score(score)
                    frame["iforest_raw"] = pd.Series(res, index=frame.index).fillna(0)
                    Console.info(f"IForest detector scored (samples={len(score)})", component="IFOREST", samples=len(score))
            
            # GMM Detector
            if gmm_enabled and gmm_detector:
                with T.section("score.gmm"):
                    res = gmm_detector.score(score)
                    frame["gmm_raw"] = pd.Series(res, index=frame.index).fillna(0)
                    Console.info(f"GMM detector scored (samples={len(score)})", component="GMM", samples=len(score))
            
            # OMR Detector (store contributions outside frame - pandas doesn't support custom attributes)
            if omr_enabled and omr_detector:
                with T.section("score.omr"):
                    omr_z, omr_contributions = omr_detector.score(score, return_contributions=True)
                    frame["omr_raw"] = pd.Series(omr_z, index=frame.index).fillna(0)
                    omr_contributions_data = omr_contributions
                    Console.info(f"OMR detector scored (samples={len(score)})", component="OMR", samples=len(score))
        
        Console.info(f"All detectors scored in {time.perf_counter()-score_start_time:.2f}s", component="MODEL")
        
        hb.stop()"""
    
    new_section = """        # Score all enabled detectors
        score_result = _score_detectors(
            score_df=score,
            detector_state=detector_state,
            ar1_detector=ar1_detector,
            pca_detector=pca_detector,
            iforest_detector=iforest_detector,
            gmm_detector=gmm_detector,
            omr_detector=omr_detector,
            T=T,
        )
        frame = score_result.frame
        omr_contributions_data = score_result.omr_contributions
        
        hb.stop()"""
    
    if old_section_start not in content:
        print("ERROR: Could not find scoring section to replace")
        return
    
    content = content.replace(old_section_start, new_section)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    new_length = len(content)
    print(f"SUCCESS: Replaced detector scoring section")
    print(f"Original: {original_length} chars, New: {new_length} chars")
    print(f"Removed: {original_length - new_length} chars")

if __name__ == "__main__":
    main()
