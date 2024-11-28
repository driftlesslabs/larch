from __future__ import annotations

default_config = """---

rules:
  braces:
    min-spaces-inside: 0
    max-spaces-inside: 0
    min-spaces-inside-empty: -1
    max-spaces-inside-empty: -1
  brackets:
    min-spaces-inside: 0
    max-spaces-inside: 0
    min-spaces-inside-empty: -1
    max-spaces-inside-empty: -1
  colons:
    max-spaces-before: 0
    max-spaces-after: 999
  commas:
    max-spaces-before: 0
    min-spaces-after: 1
    max-spaces-after: 1
  comments:
    level: warning
    require-starting-space: false
    min-spaces-from-content: 1
  comments-indentation:
    level: warning
  document-end: disable
  document-start: disable
  #  level: warning
  #  present: true
  empty-lines:
    max: 99
    max-start: 99
    max-end: 99
  empty-values:
    forbid-in-block-mappings: false
    forbid-in-flow-mappings: false
  hyphens:
    max-spaces-after: 1
  indentation:
    spaces: consistent
    indent-sequences: true
    check-multi-line-strings: false
  key-duplicates: enable
  key-ordering: disable
  line-length:
    max: 880
    allow-non-breakable-words: true
    allow-non-breakable-inline-mappings: false
  new-line-at-end-of-file: enable
  new-lines:
    type: unix
  octal-values:
    forbid-implicit-octal: false
    forbid-explicit-octal: false
  trailing-spaces: disable
  truthy: disable
  #  level: warning
"""


class Format:
    @staticmethod
    def parsable(problem, filename):
        return f"{filename}:{problem.line}:{problem.column}: [{problem.level}] {problem.message}"

    @staticmethod
    def standard(problem, filename):
        line = f"  {problem.line:d}:{problem.column:d}"
        line += max(12 - len(line), 0) * " "
        line += problem.level
        line += max(21 - len(line), 0) * " "
        line += problem.desc
        if problem.rule:
            line += f"  ({problem.rule:s})"
        return line

    @staticmethod
    def standard_color(problem, filename):
        line = f"  \033[2m{problem.line:d}:{problem.column:d}\033[0m"
        line += max(20 - len(line), 0) * " "
        if problem.level == "warning":
            line += f"\033[33m{problem.level}\033[0m"
        else:
            line += f"\033[31m{problem.level}\033[0m"
        line += max(38 - len(line), 0) * " "
        line += problem.desc
        if problem.rule:
            line += f"  \033[2m({problem.rule})\033[0m"
        return line


def yaml_check(file, config_file=None, logger=None, encoding="utf-8"):
    if logger is None:
        log = print
    else:
        log = logger.error

    try:
        from yamllint import linter
        from yamllint.config import YamlLintConfig
        from yamllint.linter import PROBLEM_LEVELS
    except ImportError:
        log(
            "yamllint is not installed, cannot inspect yaml file for formatting quality"
        )
    else:
        filepath = file[2:] if file.startswith("./") else file

        if config_file is not None:
            conf = YamlLintConfig(file=config_file)
        else:
            conf = YamlLintConfig(content=default_config)

        first = True
        any_errors = False
        max_level = 0
        with open(filepath, encoding=encoding) as f:
            for problem in linter.run(f, conf, filepath):
                if first:
                    log(f"FOUND YAML ERRORS IN {file}")
                    first = False
                    any_errors = True

                log(Format.standard(problem, file))

                max_level = max(max_level, PROBLEM_LEVELS[problem.level])

            if any_errors:
                log(f"END OF YAML ERRORS IN {file}")
