# وفاداری Top-k نسبت به مرجع پاک

این مخزن نسخهٔ تمیز و آمادهٔ انتشار کد پژوهش است. مسئلهٔ اصلی، حفظ **مجموعهٔ
بدون‌ترتیب Top-k مدل مرجع روی ورودی پاک** در پاسخ‌های مدل دفاع‌شده است. هر
نمونه‌گیری تصادفی یک query مستقل محسوب می‌شود؛ بنابراین میانگین‌گیری logits
قبل از محاسبهٔ معیار، معیار اصلی مقاله نیست.

## هشدار داوری دوسوکور

تا پایان داوری SaTML، مخزن شخصی عمومی نسازید و نام نویسنده، دانشگاه، صفحهٔ
شخصی، DOI قبلی یا badge شناسایی‌کننده اضافه نکنید. نسخهٔ فعلی عمداً با عبارت
`Anonymous Authors` آماده شده است. راهنمای مرحله‌به‌مرحله در
[docs/GITHUB_PUBLISHING_FA.md](docs/GITHUB_PUBLISHING_FA.md) آمده است.

## محتوا

- `src/topk_fidelity`: پیاده‌سازی کوچک و خوانای روش، معیارها و شمارش query؛
- `experiments/frozen`: snapshot اسکریپت‌های واقعی CIFAR-10، CIFAR-100،
  ImageNet و شاخهٔ ViT؛
- `tests`: تست‌های CPU برای هستهٔ قابل‌استفادهٔ مجدد؛
- `results`: جدول‌های کوچک نتایج فعلی؛
- `docs/REPRODUCIBILITY.md`: راهنمای بازتولید و چیدمان داده/مدل.

داده‌ها، checkpointها، candidate tensorها و خروجی‌های حجیم عمداً در Git قرار
نمی‌گیرند.

## نصب سریع

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

برای اسکریپت‌های کامل آزمایش:

```bash
python -m pip install -e ".[experiments,test]"
```

برای اجرای GPU، ابتدا نسخهٔ متناسب PyTorch/CUDA را نصب کنید.

## کنترل نسخهٔ انتشار

```bash
python -m compileall -q src experiments/frozen tests
python scripts/audit_release.py
pytest -q
sha256sum -c MANIFEST.sha256
```

شاخهٔ confirmatory مربوط به ViT هنوز در حال اجراست؛ تا زمانی که audit مرحلهٔ
`VR4` موفق نشده، نتیجهٔ آن نهایی نیست.
