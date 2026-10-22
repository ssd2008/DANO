# Dano_hackatons

Архив проектов. Каждый хакатон или проект находится в собственной ветке.

Приватный архив: среди источников есть приватные репозитории. Публикация всей коллекции раскрыла бы их содержимое.

`main` — каталог. Для кода откройте нужную ветку по ссылке в таблице. Ветки содержат полную исходную историю; истории разных проектов не объединены.

| Источник / код | Проект | Ветка | Презентация |
|---|---|---|---|
| [dano2024](https://github.com/ssd2008/Dano_hackatons/tree/codex/dano2024) | ДАНО 2024: актуальный вид кредитования клиента | `codex/dano2024` | Есть |
| [perm2024](https://github.com/ssd2008/Dano_hackatons/tree/codex/perm2024) | Пермь 2024: факторы победы игрока | `codex/perm2024` | Есть |
| [ekb2024](https://github.com/ssd2008/Dano_hackatons/tree/codex/ekb2024) | Екатеринбург 2024 | `codex/ekb2024` | Есть |
| [nn2024](https://github.com/ssd2008/Dano_hackatons/tree/codex/nn2024) | Нижний Новгород 2024 | `codex/nn2024` | **Нужно добавить** |
| [spb2024](https://github.com/ssd2008/Dano_hackatons/tree/codex/spb2024) | Санкт-Петербург 2024 | `codex/spb2024` | **Нужно добавить** |
| [DANO24](https://github.com/ssd2008/Dano_hackatons/tree/codex/DANO24) | ДАНО 2024: звонки, кредиты и matching | `codex/DANO24` | **Нужно добавить** |
| [MSK25](https://github.com/ssd2008/Dano_hackatons/tree/codex/MSK25) | Москва 2025 | `codex/MSK25` | **Нужно добавить** |
| [SPB25_ITMO](https://github.com/ssd2008/Dano_hackatons/tree/codex/SPB25_ITMO) | Санкт-Петербург / ИТМО 2025 | `codex/SPB25_ITMO` | **Нужно добавить** |
| [EKB25](https://github.com/ssd2008/Dano_hackatons/tree/codex/EKB25) | Екатеринбург 2025 | `codex/EKB25` | **Нужно добавить** |
| [NN25](https://github.com/ssd2008/Dano_hackatons/tree/codex/NN25) | Нижний Новгород 2025 | `codex/NN25` | **Нужно добавить** |
| [dano25](https://github.com/ssd2008/Dano_hackatons/tree/codex/dano25) | ДАНО 2025: анализ ДТП | `codex/dano25` | Есть — найдена локально |
| [DANO_base](https://github.com/ssd2008/Dano_hackatons/tree/codex/DANO_base) | Общая база: статистика, обработка данных и визуализация | `codex/DANO_base` | **Нужно добавить** |

## Работа локально

```bash
git clone git@github.com:ssd2008/Dano_hackatons.git
cd Dano_hackatons
git branch -r
git switch codex/dano2024
```

Для нескольких проектов одновременно используйте `git worktree`. Перед переключением сохраните локальные изменения.

## Что сохранено

- Исходные файлы и полная история коммитов, авторы и даты.
- Все исходные ветки; дополнительные ветки имеют суффикс `--<исходная-ветка>`.
- Исходные теги, если они были, — в пространстве имён `<источник>/<тег>`.
- Происхождение и SHA коммитов в [migration.json](migration.json); детали проекта — в `ARCHIVE.md` его ветки.

Issues, PR, настройки доступа и секреты Actions принадлежат старым репозиториям и не переносятся вместе с Git. Исходные репозитории сохраняются как архивы. Названия событий основаны на исходных репозиториях; места и награды не добавлялись.

Собрано 2026-10-22.
