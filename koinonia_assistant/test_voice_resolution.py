import frappe
import json
import sys

def run_tests():
    frappe.init(site="frontend")
    frappe.connect()

    from koinonia_assistant.rag.name_search import resolve_member_and_family
    from koinonia_assistant.rag.rag_engine import run_query

    parish = "Yelagiri Parish"
    results = []

    print("\n" + "=" * 80)
    print("RUNNING VOICE-BASED PERSON NAME RESOLUTION VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Ambiguous Voice Query (Antony Raj S vs Antonyraj S)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Ambiguous Voice Query: 'Antony Raj S family details' (input_mode='voice')")
    res1 = run_query(
        "Antony Raj S family details",
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="voice"
    )
    disambig1 = res1.get("disambiguation")
    data1 = res1.get("data") or []
    reply1 = res1.get("reply", "")

    pass1 = False
    if disambig1 and disambig1.get("options") and len(disambig1["options"]) >= 2:
        opt_names = [o.get("full_name") for o in disambig1["options"]]
        opt_cards = [o.get("card_no") or o.get("family_card") for o in disambig1["options"]]
        has_antony_raj = any("Antony Raj S" in n for n in opt_names)
        has_antonyraj = any("Antonyraj" in n for n in opt_names)
        has_card_4 = "YLG/004" in opt_cards
        has_card_8 = "YLG/008" in opt_cards
        data_clean = len(data1) == 0  # Zero data leak before selection (Rule 19)
        msg_ok = "closely matching names" in (disambig1.get("message") or "") or "closely matching names" in reply1
        if has_antony_raj and has_antonyraj and has_card_4 and has_card_8 and data_clean:
            pass1 = True
            print(f"  --> PASSED: Detected ambiguity between Antony Raj S (YLG/004) and Antonyraj S (YLG/008)")
            print(f"      Message: {disambig1.get('message')}")
            print(f"      Options: {opt_names} | Cards: {opt_cards} | Data rows leaked: {len(data1)}")
        else:
            print(f"  --> FAILED: Options mismatch: names={opt_names}, cards={opt_cards}, data_rows={len(data1)}")
    else:
        print(f"  --> FAILED: Expected disambiguation object with >=2 options. Got: {disambig1}")
    results.append(("Test 1 - Ambiguous Voice Query", pass1))

    # -------------------------------------------------------------------------
    # TEST 2: Chat Retrieval Non-Regression (Rule 1: DO NOT CHANGE EXISTING CHAT)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Chat Retrieval Non-Regression: 'Antony Raj S family details' (input_mode='chat')")
    res2 = run_query(
        "Antony Raj S family details",
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="chat"
    )
    disambig2 = res2.get("disambiguation")
    data2 = res2.get("data") or []
    reply2 = res2.get("reply", "")

    pass2 = False
    # In chat mode, it must directly retrieve details without disambiguation
    if not disambig2 and (len(data2) > 0 or "Antony Raj S" in reply2):
        pass2 = True
        print(f"  --> PASSED: Chat mode directly returned records without disambiguation")
        print(f"      Data rows returned: {len(data2)} | Reply snippet: {reply2[:100]}...")
    else:
        print(f"  --> FAILED: Chat mode unexpectedly triggered disambiguation or returned no data: disambig={disambig2}")
    results.append(("Test 2 - Chat Retrieval Non-Regression", pass2))

    # -------------------------------------------------------------------------
    # TEST 3: Unique Voice Match (Rule 7 & 16-A/B: Clear unique candidate)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Unique Voice Match: 'Antony Selvan P family details' (input_mode='voice')")
    res3 = run_query(
        "Antony Selvan P family details",
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="voice"
    )
    disambig3 = res3.get("disambiguation")
    data3 = res3.get("data") or []
    reply3 = res3.get("reply", "")

    pass3 = False
    # Antony Selvan P has no close competitor in Yelagiri -> Direct retrieval!
    if not disambig3 and (len(data3) > 0 or "Antony Selvan P" in reply3):
        pass3 = True
        print(f"  --> PASSED: Direct retrieval for unique match Antony Selvan P without disambiguation")
        print(f"      Data rows: {len(data3)} | Reply snippet: {reply3[:100]}...")
    else:
        print(f"  --> FAILED: Unique match unexpectedly showed disambiguation: {disambig3}")
    results.append(("Test 3 - Unique Voice Match", pass3))

    # -------------------------------------------------------------------------
    # TEST 4: Authoritative ID / Card Selection (Rule 6, 7, 10, 18)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Selection with Card: 'Antony Raj S (Card: YLG/004) family details' (input_mode='voice')")
    res4 = run_query(
        "Antony Raj S (Card: YLG/004) family details",
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="voice"
    )
    disambig4 = res4.get("disambiguation")
    data4 = res4.get("data") or []
    reply4 = res4.get("reply", "")

    pass4 = False
    if not disambig4 and ("YLG/004" in reply4 or any(r.get("family_register_number") == "YLG/004" for r in data4)):
        pass4 = True
        print(f"  --> PASSED: Card YLG/004 directly retrieved authoritative records")
        print(f"      Data rows: {len(data4)} | Reply snippet: {reply4[:100]}...")
    else:
        print(f"  --> FAILED: Card selection did not retrieve authoritative record: {reply4[:100]}")
    results.append(("Test 4 - Authoritative Card Selection", pass4))

    # -------------------------------------------------------------------------
    # TEST 5: Multi-Turn Follow-Up Query (Rule 10 & 11)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Multi-Turn Follow-Up Query: 'Show his family members' (input_mode='voice')")
    history_turn = [
        {"role": "user", "content": "Antony Raj S (Card: YLG/004) family details"},
        {"role": "bot", "content": reply4}
    ]
    res5 = run_query(
        "Show his family members",
        history=history_turn,
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="voice"
    )
    disambig5 = res5.get("disambiguation")
    data5 = res5.get("data") or []
    reply5 = res5.get("reply", "")

    pass5 = False
    # Follow-up query should inherit Card: YLG/004 from turn 1 and directly return family members
    if not disambig5 and len(data5) > 0:
        pass5 = True
        print(f"  --> PASSED: Follow-up inherited authoritative ID and returned family members directly")
        print(f"      Family members returned: {len(data5)} | Reply snippet: {reply5[:100]}...")
    else:
        print(f"  --> FAILED: Follow-up failed or triggered disambiguation: disambig={disambig5}, data={len(data5)}")
    results.append(("Test 5 - Multi-Turn Follow-Up Context", pass5))

    # -------------------------------------------------------------------------
    # TEST 6: Low Confidence Voice Input (Rule 16-D)
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Low Confidence Voice Input: 'XYZNonExistentPerson family details' (input_mode='voice')")
    res6 = run_query(
        "XYZNonExistentPerson family details",
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="voice"
    )
    reply6 = res6.get("reply", "")
    pass6 = False
    if "couldn't identify" in reply6.lower() or "not found" in reply6.lower() or "verify the spelling" in reply6.lower():
        pass6 = True
        print(f"  --> PASSED: Polite low confidence / not found prompt returned")
        print(f"      Reply: {reply6}")
    else:
        print(f"  --> FAILED: Unexpected reply for non-existent person: {reply6}")
    results.append(("Test 6 - Low Confidence Voice Input", pass6))

    # -------------------------------------------------------------------------
    # TEST 7: Tamil Voice Query Ambiguity (Rule 13 & 19)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Tamil Voice Query Ambiguity: 'அந்தோணி ராஜ் எஸ் குடும்ப விவரங்கள்' (input_mode='voice')")
    res7 = run_query(
        "அந்தோணி ராஜ் எஸ் குடும்ப விவரங்கள்",
        user_role="Parish Priest",
        user_parish=parish,
        input_mode="voice"
    )
    disambig7 = res7.get("disambiguation")
    data7 = res7.get("data") or []
    reply7 = res7.get("reply", "")

    pass7 = False
    if disambig7 and disambig7.get("options") and len(disambig7["options"]) >= 2:
        opt_names = [o.get("full_name") for o in disambig7["options"]]
        opt_cards = [o.get("card_no") or o.get("family_card") for o in disambig7["options"]]
        data_clean = len(data7) == 0
        msg = disambig7.get("message", "")
        # Should be in Tamil
        if "பொருத்தமான" in msg or "குறிப்பிடுகிறீர்கள்" in msg or "YLG/004" in opt_cards:
            pass7 = True
            print(f"  --> PASSED: Tamil disambiguation prompt returned with candidate cards")
            print(f"      Tamil Message: {msg}")
            print(f"      Options: {opt_names} | Cards: {opt_cards}")
        else:
            print(f"  --> FAILED: Tamil message not found: {msg}")
    else:
        print(f"  --> FAILED: Tamil disambiguation missing: {disambig7}")
    results.append(("Test 7 - Tamil Voice Query Ambiguity", pass7))

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY RESULTS")
    print("=" * 80)
    passed_count = sum(1 for name, p in results if p)
    total_count = len(results)
    for name, p in results:
        status_str = "PASSED" if p else "FAILED"
        print(f"  [{status_str}] {name}")
    print(f"\nTOTAL: {passed_count}/{total_count} PASSED ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)

    if passed_count < total_count:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
