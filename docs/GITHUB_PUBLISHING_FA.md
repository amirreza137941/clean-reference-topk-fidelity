# راهنمای انتشار مخزن در GitHub

این راهنما برای همین بسته نوشته شده است. پیشنهاد امن برای دورهٔ داوری SaTML
این است که ابتدا یک مخزن **Private** در حساب خود بسازید، نسخهٔ ناشناس را از آن
آماده کنید، و مخزن شخصی را تا پایان تصمیم داوری عمومی نکنید.

## ۱. قبل از بارگذاری

پوشهٔ ZIP را باز کنید و در ریشهٔ پروژه این کنترل‌ها را اجرا کنید:

```bash
python -m compileall -q src experiments/frozen tests
python scripts/audit_release.py
```

اگر محیط شما PyTorch و وابستگی تست را دارد:

```bash
python -m pip install -e ".[test]"
pytest -q
```

موارد زیر نباید وارد Git شوند:

- `data/`، `datasets/`، `models/`، `checkpoints/` و `outputs/`؛
- فایل‌های `*.pth`، `*.pt`، `*.npz` و candidate tensorها؛
- توکن، کلید SSH، لینک امضاشدهٔ فضای ابری یا مسیر واقعی سرور؛
- نام نویسندگان، دانشگاه، acknowledgement، صفحهٔ شخصی و ORCID در نسخهٔ داوری.

## ۲. ساخت مخزن در GitHub

در GitHub روی **New repository** بزنید و این مقادیر را بگذارید:

| فیلد | مقدار پیشنهادی |
|---|---|
| Repository name | `clean-reference-topk-fidelity` |
| Description | `Code for evaluating clean-reference exact Top-k set fidelity under adversarial queries.` |
| Visibility | `Private` تا پایان داوری |
| Initialize with README | خاموش |
| Add .gitignore | `None` |
| Choose a license | `None` |

README، `.gitignore` و LICENSE از قبل در بسته هستند. GitHub نیز توصیه می‌کند
هنگام واردکردن یک پروژهٔ محلی، مخزن مقصد را با README/license/gitignore جدید
پیش‌پر نکنید تا conflict اولیه ایجاد نشود:
[راهنمای رسمی ساخت مخزن](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).

## ۳. فرستادن کد با Git

ترمینال را در پوشهٔ بازشدهٔ پروژه باز کنید. به‌جای `<USERNAME>` نام حساب خود
را بگذارید:

```bash
git init -b main
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.invalid"
git add .
git status
git commit -m "Initial anonymized research release"
git remote add origin https://github.com/<USERNAME>/clean-reference-topk-fidelity.git
git push -u origin main
```

دو دستور `git config` فقط روی همین مخزن اثر می‌گذارند و مانع درج نام/ایمیل
واقعی در metadata کامیت داوری می‌شوند. حسابی که به مخزن Private push می‌کند
طبعاً برای خود GitHub معلوم است؛ آن مخزن را عمومی یا مستقیماً به مقاله لینک
نکنید و برای داور فقط mirror ناشناس را ارائه دهید.

پیش از `git commit` خروجی `git status` را بخوانید. نباید dataset، checkpoint،
نتایج خام حجیم یا فایل خارج از ساختار این بسته دیده شود.

اگر برای GitHub احراز هویت HTTPS می‌خواهد، از Personal Access Token یا GitHub
CLI استفاده کنید؛ رمز معمول حساب برای عملیات Git پذیرفته نمی‌شود. روش جایگزین
با GitHub CLI:

```bash
gh auth login
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.invalid"
gh repo create clean-reference-topk-fidelity --private --source=. --remote=origin --push
```

یکی از دو روش را اجرا کنید، نه هر دو.

## ۴. آماده‌سازی صفحهٔ مخزن

در سمت راست صفحه، کنار **About** روی چرخ‌دنده بزنید:

- Description:
  `Code for evaluating clean-reference exact Top-k set fidelity under adversarial queries.`
- Website: در دورهٔ داوری خالی؛ بعداً صفحهٔ مقاله یا DOI artifact؛
- Topics:
  `adversarial-machine-learning`, `top-k`, `model-robustness`,
  `black-box-attacks`, `pytorch`, `reproducibility`؛
- Releases و Packages را فقط در صورت استفاده فعال نگه دارید.

فایل `README.md` خود صفحهٔ اصلی را می‌سازد و شامل معرفی، فرمول، نصب، ساختار
مخزن، وضعیت نتیجهٔ ViT و مسیر بازتولید است. تصویر روش نیز محلی است و وابستگی
به سایت خارجی ندارد.

در **Settings → General**:

- Visibility را Private نگه دارید؛
- Wikis را فعلاً خاموش کنید تا مستندات دو نسخه نشوند؛
- Issues را فقط اگر کانال anonymous مناسبی دارید فعال کنید؛
- در بخش Pull Requests، حذف خودکار branch پس از merge را می‌توانید روشن کنید.

در **Settings → Branches** برای `main` یک protection rule بسازید و دست‌کم
اجرای موفق CI پیش از merge را لازم کنید. workflow موجود، audit انتشار و تست‌ها
را اجرا می‌کند.

## ۵. نسخهٔ ناشناس برای SaTML

SaTML 2027 داوری دوسوکور دارد و صریحاً می‌گوید artifact و سایر مواد تکمیلی
باید anonymized باشند؛ حتی اشارهٔ قابل‌شناسایی به پروژهٔ open-source عمومی
می‌تواند هویت را آشکار کند. در فراخوان رسمی، مهلت artifact برابر **۲ اکتبر
۲۰۲۶** ذکر شده و استفاده از سرویسی مانند `anonymous.4open.science` پیشنهاد شده
است. نسخهٔ ارسالی در دورهٔ داوری نباید تغییر کند:
[SaTML 2027 Call for Papers](https://satml.org/call-for-papers/).

روال پیشنهادی:

1. این مخزن شخصی را Private نگه دارید؛
2. `python scripts/audit_release.py` را اجرا کنید؛
3. یک commit مشخص را برای artifact انتخاب کنید؛
4. از همان commit یک mirror ناشناس و فقط‌خواندنی بسازید؛
5. لینک ناشناس را در سامانهٔ ارسال مقاله قرار دهید؛
6. بعد از freeze، همان artifact را در طول داوری ویرایش نکنید.

از fork عمومی، star کردن با حساب شخصی، issue با نام واقعی، badge حساب شخصی،
یا لینک مستقیم از صفحهٔ دانشگاه به artifact ناشناس خودداری کنید.

## ۶. افزودن نتیجهٔ ViT

پس از اتمام VR4، تنها در صورتی نتیجه را اضافه کنید که self-test، finalizer و
audit همان stage موفق باشند. پیشنهاد commit:

```bash
git checkout -b results/vit-vr4
# نتیجهٔ کوچک و audit report را اضافه کنید؛ raw tensors را اضافه نکنید.
python scripts/audit_release.py
git add results README.md experiments/frozen
git commit -m "Add audited ViT VR4 confirmatory results"
git push -u origin results/vit-vr4
```

در دورهٔ freeze داوری، این تغییر باید برای نسخهٔ بعد از داوری بماند، مگر آنکه
سیاست رسمی venue صریحاً اجازهٔ update بدهد.

## ۷. عمومی‌کردن پس از تصمیم داوری

بعد از رفع محدودیت دوسوکور:

1. `Anonymous Authors` را با نام نویسندگان جایگزین کنید؛
2. `CITATION.cff.template` را تکمیل و به `CITATION.cff` تغییر نام دهید؛
3. URLهای مقاله، artifact و مستندات را اضافه کنید؛
4. audit و test را دوباره اجرا کنید؛
5. visibility را Public کنید؛
6. tag و release بسازید:

```bash
git tag -a v1.0.0 -m "Artifact release v1.0.0"
git push origin v1.0.0
```

سپس از صفحهٔ **Releases → Draft a new release** همان tag را انتخاب و changelog
کوتاه اضافه کنید. راهنمای رسمی:
[مدیریت Releaseها در GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

برای نسخهٔ archival، release نهایی را به Zenodo متصل و DOI ثابت را در README و
CITATION ثبت کنید. خود فراخوان SaTML نیز برای مقالهٔ پذیرفته‌شده نگهداری artifact
در مخزنی مانند Zenodo را توصیه می‌کند.

## چک نهایی ۳۰ ثانیه‌ای

- مخزن شخصی هنوز Private است؛
- صفحهٔ README بدون نام و affiliation است؛
- `scripts/audit_release.py` با exit code صفر تمام می‌شود؛
- هیچ داده، checkpoint یا secret در `git status` نیست؛
- نتیجهٔ ViT تا موفقیت audit با برچسب `pending` مانده است؛
- لینک مقاله فقط به نسخهٔ ناشناس و frozen اشاره می‌کند.
