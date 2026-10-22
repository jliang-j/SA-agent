from fastapi.testclient import TestClient

import sa_agent.main as main


def test_frontend_assets():
    """页面及静态资源可用，且原有健康检查继续响应。"""
    with TestClient(main.app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert 'id="research-form"' in page.text
        for name, media in (("app.js", "javascript"), ("style.css", "text/css")):
            response = client.get(f"/static/{name}")
            assert response.status_code == 200
            assert media in response.headers["content-type"]
        assert client.get("/health").json() == {"status": "ok"}
