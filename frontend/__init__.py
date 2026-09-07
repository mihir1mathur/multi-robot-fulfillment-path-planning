"""Streamlit demonstration dashboard for the fulfillment system.

This package is a THIN CLIENT. It contains no planning, allocation,
coordination or recovery logic of its own - every computation is done by the
FastAPI service over HTTP. The modules here only:

    api_client.py   a small typed HTTP client for the service
    view_model.py   pure functions: API responses -> tables / grid / events
    scenario.py     the fixed demo scenario (a warehouse + robots + tasks)
    app.py          the Streamlit UI

Run it with:

    uvicorn robotics.api.app:create_app --factory      # terminal 1
    streamlit run frontend/app.py                       # terminal 2
"""
