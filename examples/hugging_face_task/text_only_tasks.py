"""Shared task filters for APEX text-only benchmark runs.

Default policy is the AA conservative text-only subset: keep the normal
AA-strict 452-task set, then exclude tasks classified as definitely requiring
vision plus tasks whose attached artifacts contain embedded images/charts/media
that may be risky for text-only models. That yields 420 tasks.

The broader semantic text-only policy excludes only definitely vision-required
tasks and yields 445 tasks.
"""

AA_EXCLUDED_TASK_IDS = frozenset(
    {
        "task_052cc6311cf34bc6bacc4f521ba77460",
        "task_0ffa0e6ff7d3433e98582e50e79068cd",
        "task_15c7a39c67a14b11862f157ec6197f40",
        "task_16c0324b442841ec86f8ae24cbde119e",
        "task_1f84a712cb2e4aaaa4b6778eeff49021",
        "task_1fb84d7682dc43138ad220b203ed5b22",
        "task_278eac61c4ee4155a75744086715a0e8",
        "task_39c68b482c08464c8fb06cd5af932cd6",
        "task_4c709105f6f649dcbe6fe98bd71dad32",
        "task_5a7117ac62fd4da9bec41fe8d805ee03",
        "task_68a8fbc9544640cf9a20db80dd845d85",
        "task_6c4429d4d63f46cdbc87b09a4bd75d2f",
        "task_6fa5694c8bbe434e944d76e6782369b1",
        "task_7063e8d0a91e4e74a5b7f40952af917e",
        "task_754401fc583e449bafb8bdcd61f927e3",
        "task_761646f23fbb4a0b8564e4b348d14de1",
        "task_7937759836244ed4a9cfb65c70e0e746",
        "task_7c394865481b40cdbdd577a039825679",
        "task_7d11f0f8a4ac415599f715647d2a09e4",
        "task_83bed0e08f1b45efb40ad8a64deb6fd8",
        "task_883f8bcbf38148648037f16db02a9754",
        "task_91b0998a0b23403d9aeb1c10f75a22b1",
        "task_9a7eb18bc7084c22a4d96d9818faeaa4",
        "task_a006f24d413c4dc99dd644d5e0dc12f7",
        "task_b8270cca4f7c455791d7b9807ed34295",
        "task_c917c8e632364886af9a2fc1ee95d4ca",
        "task_c99cdf2356174ea8a0fc7a4f9b4e95f4",
        "task_fc51bd4130bf475faa36a5d45a96adb3",
    }
)

VISION_REQUIRED_TASK_IDS = frozenset(
    {
        "task_0233800d9daf4459bc464ce2f1f822a8",
        "task_0a4ad19b76cf4602914e6b8a4f263690",
        "task_749eeedb6a2b4a8a98ddd46fed5ac7b7",
        "task_8bd0a44c2c154bb3a089f5f19c9e0a23",
        "task_be688e67984c42c9bca2df799f50a31a",
        "task_f23cb148241641f1b7c5dfbecfd3835f",
        "task_f69f5d19990b4292809009a331e1bbe9",
    }
)

VISION_LIKELY_OR_RISKY_TASK_IDS = frozenset(
    {
        "task_0c527e11cc1c436696561a15eead68b2",
        "task_11d91f7c17424faa8f89a5a46c47b76f",
        "task_18482ca6de9943ce814d70f2f742497f",
        "task_2b2666310e7e4712be0f2c0e4240d5a2",
        "task_2bd66e1e194a4ce89ccf6432cbdce451",
        "task_4372ee27c60f4589884be8cc9d6d8bd8",
        "task_4e1c98d2f8a24bbaa1fbd08dc51c04a2",
        "task_626333c69ffb46d0ad041a2dd6916fdf",
        "task_88469a82ab624b0084d2381fbaeb38d1",
        "task_8d7835ee0dce4b92b89ebe15540be78c",
        "task_9363f85adf864ccaa8dbff72d38225cb",
        "task_953125d9b5634c68acffd075acf47448",
        "task_976f61c9753f494a9ad012af60b3309c",
        "task_a1449f78d0c3427e9b34e65a08621976",
        "task_a179d38b095f46eba5eff7baf8f7fd87",
        "task_a89f67b98b5e468d8d5f2a359db895d6",
        "task_aac22560bcdc434eb7942bce0631d8bb",
        "task_b63eb63a9b964203bb7033ed3682f06e",
        "task_cb687edb826643a7857ab1f4cde2c0a5",
        "task_d55fe268d7f64a74aacfce4fc374ea96",
        "task_ded0b246614049ab85ad985d45e44a30",
        "task_e75cacb35dc8429a895bba6aff5f8a58",
        "task_f029be9cd145432599e5627d4111af24",
        "task_f14e5c8e67ba4b018f537c990ea96d71",
        "task_f18f9e5701bf47a6a835d7d7bd6c7024",
    }
)

FULL_DATASET_TASK_COUNT = 480
AA_STRICT_TASK_COUNT = 452
AA_TEXT_ONLY_TASK_COUNT = 445
AA_CONSERVATIVE_TEXT_ONLY_TASK_COUNT = 420

# Backward-compatible names used by older scripts. These follow the default
# conservative text-only policy.
VISION_EXCLUDED_TASK_IDS = VISION_REQUIRED_TASK_IDS | VISION_LIKELY_OR_RISKY_TASK_IDS
TEXT_ONLY_TASK_COUNT = AA_CONSERVATIVE_TEXT_ONLY_TASK_COUNT


def is_aa_strict_task(task_id: str) -> bool:
    return task_id not in AA_EXCLUDED_TASK_IDS


def is_text_only_task(
    task_id: str,
    *,
    include_aa_excluded: bool = False,
    conservative: bool = True,
) -> bool:
    if not include_aa_excluded and task_id in AA_EXCLUDED_TASK_IDS:
        return False
    if task_id in VISION_REQUIRED_TASK_IDS:
        return False
    if conservative and task_id in VISION_LIKELY_OR_RISKY_TASK_IDS:
        return False
    return True


def filter_text_only_tasks(
    tasks: list[dict],
    *,
    include_aa_excluded: bool = False,
    conservative: bool = True,
) -> list[dict]:
    return [
        task
        for task in tasks
        if is_text_only_task(
            task["task_id"],
            include_aa_excluded=include_aa_excluded,
            conservative=conservative,
        )
    ]
