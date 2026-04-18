// ==========================================
// i18n.js — منطق التبديل بين العربي والإنجليزي
// ==========================================

function setLanguage(lang) {
  localStorage.setItem("lang", lang);

  // تغيير اتجاه الصفحة
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";

  // ترجمة النصوص العادية
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (translations[lang] && translations[lang][key]) {
      el.textContent = translations[lang][key];
    }
  });

  // ترجمة الـ placeholder
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (translations[lang] && translations[lang][key]) {
      el.placeholder = translations[lang][key];
    }
  });

  // تحديث زر اللغة
  const langBtn = document.querySelector(".language-selector");
  if (langBtn) {
    langBtn.textContent = lang === "ar" ? "EN" : "AR";
  }

  // تعديل font للعربي
  document.body.style.fontFamily = lang === "ar"
    ? "'Segoe UI', 'Arial', sans-serif"
    : "Arial, sans-serif";
}

function toggleLanguage() {
  const current = localStorage.getItem("lang") || "en";
  setLanguage(current === "en" ? "ar" : "en");
}

// تحميل اللغة المحفوظة عند فتح الصفحة
document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("lang") || "en";
  setLanguage(saved);
});
