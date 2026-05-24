from app.services.rrf import reciprocal_rank_fusion


def test_rrf_merges_by_article_code():
    fused = reciprocal_rank_fusion([
        [{"article_code": "12", "title": "A", "score": 2, "source": "x"}],
        [{"article_code": "12", "title": "A", "score": 1, "source": "y"}, {"article_code": "17", "title": "B", "score": 1, "source": "y"}],
    ])
    assert fused[0]["article_code"] == "12"
    assert "x" in fused[0]["sources"]
    assert "y" in fused[0]["sources"]
