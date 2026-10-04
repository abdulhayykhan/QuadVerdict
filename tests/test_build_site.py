"""
test_build_site.py — Tests for web/build_site.py

Assertions (Phase 8, to be filled in):
1. Placeholder token /*__RESULTS_JSON__*/ is replaced in dist/index.html output.
2. Injected JSON round-trips: json.loads(extracted_json) == original_dict.
3. No "</script>" string in the injected JSON (escape test: </ becomes <\\/  in JSON).
4. Output file contains <html>, <head>, <body> (basic HTML parse check).
5. build_site.py --mock produces a file that opens via file:// (size < DIST_MAX_MB).
6. Placeholder appearing 0 or 2+ times → ValueError (build should fail fast).
"""
# Tests will be implemented in Phase 8.
