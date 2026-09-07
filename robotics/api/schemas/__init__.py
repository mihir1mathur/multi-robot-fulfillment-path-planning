"""Pydantic request / response models - the API's typed contract.

These are DISTINCT from the SQLAlchemy ORM models in
``robotics.persistence.models``:

    ORM model    = how a row is stored (columns, keys, indexes)
    API schema   = what a client may send / will receive (validation, shape)

Keeping them separate means the database can change columns without changing
the public contract, and the API can validate / rename fields without touching
storage. A thin mapping layer in each router converts between them.
"""
