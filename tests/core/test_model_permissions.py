from src.core.model_permissions import check_model_allowed_with_aliases


class TestCheckModelAllowedWithAliases:
    def test_exact_match_returns_allowed_without_mapping(self) -> None:
        is_allowed, matched = check_model_allowed_with_aliases(
            model_name="gpt-4o",
            allowed_models=["gpt-4o"],
            api_format="OPENAI",
            resolved_model_name="gpt-4o",
            model_aliases=[r"gpt-4o-.*"],
        )
        assert is_allowed is True
        assert matched is None

    def test_alias_match_is_deterministic(self) -> None:
        is_allowed, matched = check_model_allowed_with_aliases(
            model_name="target",
            allowed_models=["b", "a"],
            api_format="OPENAI",
            resolved_model_name="target",
            model_aliases=[r".*"],
        )
        assert is_allowed is True
        assert matched == "a"

    def test_alias_match_respects_candidate_models(self) -> None:
        is_allowed, matched = check_model_allowed_with_aliases(
            model_name="target",
            allowed_models=["other-1", "allowed-1"],
            api_format="OPENAI",
            resolved_model_name="target",
            model_aliases=[r".*-1"],
            candidate_models={"allowed-1"},
        )
        assert is_allowed is True
        assert matched == "allowed-1"

    def test_alias_match_candidate_models_no_intersection(self) -> None:
        is_allowed, matched = check_model_allowed_with_aliases(
            model_name="target",
            allowed_models=["allowed-1"],
            api_format="OPENAI",
            resolved_model_name="target",
            model_aliases=[r".*-1"],
            candidate_models={"not-present"},
        )
        assert is_allowed is False
        assert matched is None

