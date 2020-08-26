Changes that were made to original **PPBO** project:

The files below were changed in the same manner:

- `Camphor_Copper/create_111_camphor_func.py`
- `Camphor_Copper/GUI.py`

From:

```python
path_from_root_to_files = os.getcwd() + '/Camphor_Copper/'
```

To:

```python
path_from_root_to_files = os.getcwd() + '/PPBO/Camphor_Copper/'
```

Other changes:

- `ppbo_settings.py`
    - Several parameters are added as a parameter to `__init__` of `PPBO_settings` class (and to GANPrefFinder):
        - `max_iter_fMAP_estimation`
        - `mu_star_finding_trials`





