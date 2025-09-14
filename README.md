# serenity

## Environment Variables

- `OPENAI_HYDE_MODEL`: Optional. Model used for generating the hypothetical answer in the HyDE search step. Defaults to `gpt-3.5-turbo` if not set. This can also be specified via the `hyde_model` parameter when calling `backend.generative_search`.
- `AIRTABLE_REPORTS_TABLE_NAME`: Optional. Airtable table where pre-generated reports are stored. Defaults to `GeneratedReports`.
=======
## Running Tests

Install dependencies and run the test suite with:

```
pip install -r requirements.txt
pytest
```
