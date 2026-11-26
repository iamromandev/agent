# Data model

Every Tortoise model, column and enum is documented here. There are none yet.

## Tortoise ORM notation

- Models subclass `IdBase` → `LinkBase` → `Base` (`src/core/base.py`); each rung adds bookkeeping the one below does without.
- Uniqueness rules live in `Meta.constraints` as `UniqueConstraint` (partial uniques carry `condition=`); plain uniques use `unique_together`.
- Tortoise 1.1.8 renders only `unique_together` and `Meta.indexes` in `CreateModel`, so a constraint on a new model needs an explicit `AddConstraint` op in its migration.
