![happyml](../../happyml.png)

# SET (future)
[Back to the table of contents](../README.md)

Disclaimer: happyml is constantly changing. If I see the chance for improvement, I'm going to take it. I hope that you get value out of it,
but don't be shocked if you see changes. I'm trying to make it better, and if I find something doesn't feel right or good to me, I won't hessitate
to rewrite how it works. I hope you can appreciate the spirit of this project.

## Syntax

```happyml
set <property> <value>
```

## Description and Notes
**This is a future feature.**

Sets a global happyml property to a value.

Valid properties:
* `output` -- Sets the output destination. Valid values are:
  * `console` -- Prints output to the console. **Default if not specified.**
  * `file` -- Prints output to a file.
* `output_file` -- Sets the output file path. Only valid if `output` is set to `file`.
* `output_format` -- Sets the output format. Valid values are:
  * `parsable` -- Prints output in a parsable format.
  * `human` -- Prints output in a human-readable format. **Default if not specified.**
* `output_mode` -- Sets the output mode. Valid values are:
  * `append` -- Appends output to the output file. **Default if not specified.**
  * `overwrite` -- Overwrites the output file.
* `token_encoder_path` -- Sets the path to the byte pair encoder model.
* `repo_path` -- Sets the path to the happyml repository. **Defaults to the ..\happyml_repo directory.**
* `data_path` -- Sets the path to the happyml data directory. **Defaults to the ..\happyml_data directory.**

## Example

```happyml
set output_format parsable
```
