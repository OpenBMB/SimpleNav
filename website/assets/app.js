const state = {
  lang: localStorage.getItem("simplenav-lang") || "en",
  data: null,
  heroPage: 0,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const copy = (value) => typeof value === "string" ? value : value?.[state.lang] ?? value?.en ?? "";

function setText(element, value) {
  if (!element) return;
  element.innerHTML = copy(value);
}

function renderStaticCopy() {
  $$('[data-copy]').forEach((element) => {
    const value = element.dataset.copy.split(".").reduce((acc, key) => acc?.[key], state.data);
    setText(element, value);
  });
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.title = state.lang === "zh" ? "SimpleNav: 让导航 VLA 变得简单" : "SimpleNav: Make Navigation VLA Simple";
  $$('[data-nav]').forEach((link) => { link.textContent = copy(state.data.nav[link.dataset.nav]); });
  $("[data-readme-link]").href = `https://github.com/fulanya55/starVLA/blob/SimpleNav/${state.lang === "zh" ? "README_ZH.md" : "README.md"}`;
  const frameworkImage = $("[data-framework-image]");
  if (frameworkImage) {
    frameworkImage.src = state.lang === "zh"
      ? "assets/figures/simplenav_framework_zh.png"
      : "assets/figures/simplenav_framework.png";
    frameworkImage.alt = state.lang === "zh"
      ? "SimpleNAV 数据转换、模型训练和闭环测评整体框架"
      : "SimpleNAV data conversion, model training, and closed-loop evaluation framework";
  }
  const languageButton = $("[data-language]");
  if (languageButton) languageButton.textContent = state.lang === "en" ? "中文" : "EN";
}

function renderFramework(active = "data") {
  const nodes = state.data.frameworkNodes;
  $$("[data-framework]").forEach((button) => {
    const selected = button.dataset.framework === active;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    const item = nodes[button.dataset.framework];
    if (item) {
      $("strong", button).textContent = copy(item.label);
      $("small", button).textContent = copy(item.subtitle);
    }
  });
  const item = nodes[active];
  $("#framework-detail").innerHTML = `
    <div><div class="eyebrow">${copy(item.label).toUpperCase()}</div><h3>${copy(item.title)}</h3><p>${copy(item.text)}</p></div>
    <ul class="detail-list">${item.items.map((value) => `<li>${copy(value)}</li>`).join("")}</ul>`;
}

function renderModels(active = "backbone") {
  const container = $("#model-tabs");
  container.innerHTML = state.data.modelCards.map((item) => `<button class="model-tab ${item.id === active ? "active" : ""}" type="button" data-model="${item.id}" role="tab">${copy(item.label)}</button>`).join("");
  const item = state.data.modelCards.find((entry) => entry.id === active) || state.data.modelCards[0];
  $("#model-detail").innerHTML = `<div class="eyebrow">${copy(item.label).toUpperCase()}</div><h3>${copy(item.title)}</h3><p>${copy(item.text)}</p><div class="tag-list">${item.tags.map((tag) => `<span class="tag">${tag}</span>`).join("")}</div>`;
  $$("[data-model]", container).forEach((button) => button.addEventListener("click", () => renderModels(button.dataset.model)));
}

function renderData() {
  $("#data-datasets").innerHTML = state.data.data.datasets.map((item) => `
    <span class="data-dataset"><strong>${item.name}</strong><span> · ${item.platform}</span></span>`).join("");
}

function pauseOtherVideos(current) {
  $$("video").forEach((video) => { if (video !== current) video.pause(); });
}

function renderHeroCarousel(page = state.heroPage) {
  const pageSize = 3;
  const items = state.data.heroDemoIds
    .map((id) => state.data.demosList.find((item) => item.id === id))
    .filter(Boolean);
  const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
  state.heroPage = (page + pageCount) % pageCount;
  const visible = items.slice(state.heroPage * pageSize, state.heroPage * pageSize + pageSize);
  $("#hero-carousel").innerHTML = visible.map((item) => `
    <article class="hero-demo-card">
      <video controls preload="none" playsinline poster="${item.poster}">
        <source src="${item.video}" type="video/mp4">
      </video>
      <div><strong>${item.title}</strong><span>${copy(item.subtitle)}</span></div>
    </article>`).join("");
  $("#hero-carousel-status").textContent = `${state.heroPage + 1} / ${pageCount}`;
  $$('[data-hero-direction]').forEach((button) => {
    button.setAttribute("aria-label", copy(state.data.actions[button.dataset.heroDirection === "previous" ? "previous" : "next"]));
    button.onclick = () => renderHeroCarousel(state.heroPage + (button.dataset.heroDirection === "previous" ? -1 : 1));
  });
  $$(".hero-demo-card video").forEach((video) => video.addEventListener("play", () => pauseOtherVideos(video)));
}

function renderBenchmarks(active = "openfly", activeView = null) {
  const list = state.data.benchmarksList;
  $("#benchmark-switcher").innerHTML = list.map((item) => `<button class="benchmark-tab ${item.id === active ? "active" : ""}" type="button" data-benchmark="${item.id}" role="tab">${copy(item.label)}</button>`).join("");
  const item = list.find((entry) => entry.id === active) || list[0];
  const view = item.views.find((entry) => entry.id === activeView) || item.views[0];
  const viewSwitcher = item.views.length > 1
    ? `<div class="benchmark-view-switcher" role="tablist" aria-label="${state.lang === "zh" ? "数据子集或任务" : "Split or task"}">${item.views.map((entry) => `<button class="benchmark-view-tab ${entry.id === view.id ? "active" : ""}" type="button" data-benchmark-view="${entry.id}" role="tab">${copy(entry.label)}</button>`).join("")}</div>`
    : "";
  const inputLabel = state.lang === "zh" ? "输入" : "Input";
  const artifactLabel = state.lang === "zh" ? "来源" : "Source";
  const trainingLabel = state.lang === "zh" ? "训练数据量" : "Training data";
  const modelSizeLabel = state.lang === "zh" ? "模型大小" : "Model size";
  const metricNumber = (value) => {
    const match = String(value).match(/-?\d+(?:\.\d+)?/);
    return match ? Number(match[0]) : null;
  };
  const sotaValues = view.columns.map((column, index) => {
    const values = view.rows
      .filter((row) => copy(row.method).toLowerCase() !== "human")
      .map((row) => metricNumber(row.values[index]))
      .filter((value) => value !== null);
    if (!values.length) return null;
    return column.includes("↓") ? Math.min(...values) : Math.max(...values);
  });
  const comparisonRows = view.rows.map((row) => {
    const methodKey = row.highlight ? "__ours" : copy(row.method);
    const artifact = state.data.artifactProfileOverrides[item.id]?.[methodKey] || state.data.artifactProfiles[methodKey];
    const artifactText = artifact?.url
      ? `<a href="${artifact.url}" target="_blank" rel="noreferrer">${artifactLabel} · ${copy(artifact.label)} ↗</a>`
      : artifact ? `<span>${artifactLabel} · ${copy(artifact.label)}</span>` : "";
    const training = state.data.trainingDataProfiles[item.id]?.[methodKey];
    const modelSize = state.data.modelSizeProfiles?.[item.id]?.[methodKey];
    const metadata = [
      row.input ? `<span>${inputLabel} · ${copy(row.input)}</span>` : "",
      training ? `<span>${trainingLabel} · ${copy(training)}</span>` : "",
      modelSize ? `<span>${modelSizeLabel} · ${copy(modelSize)}</span>` : "",
      artifactText,
    ].filter(Boolean).join("");
    const rowSota = view.columns.map((column, index) => {
      const value = metricNumber(row.values[index]);
      return value !== null && sotaValues[index] !== null && value === sotaValues[index];
    });
    const methodClass = rowSota.some(Boolean) ? "sota" : "";
    const values = view.columns.map((column, index) => `<div class="comparison-value"><span>${column}</span><strong class="${rowSota[index] ? "sota" : ""}">${row.values[index]}</strong></div>`).join("");
    return `<div class="comparison-row ${row.highlight ? "highlight" : ""}"><div class="comparison-method"><strong class="${methodClass}">${copy(row.method)}</strong><div class="comparison-meta">${metadata}</div></div><div class="comparison-values" style="--metric-count: ${view.columns.length}">${values}</div></div>`;
  }).join("");
  $("#benchmark-panel").innerHTML = `
    <div class="benchmark-top"><div><h3>${copy(item.label)}</h3><span class="benchmark-status">${copy(item.status)}</span></div><a class="benchmark-source" href="${copy(item.source)}" target="_blank" rel="noreferrer">${state.lang === "zh" ? "查看完整测评文档" : "Open full benchmark doc"} ↗</a></div>
    ${viewSwitcher}
    <div class="benchmark-view-heading"><div><span>${state.lang === "zh" ? "当前结果" : "Current result"}</span><h4>${copy(view.label)}</h4></div><p>${copy(view.note)}</p></div>
    <div class="metrics-grid">${view.metrics.map((metric) => `<div class="metric"><span class="metric-label">${metric.label}</span><strong class="metric-value">${metric.value}</strong></div>`).join("")}</div>
    <p class="benchmark-description">${copy(item.description)}</p>
    <div class="comparison-heading"><span>${state.lang === "zh" ? "方法对比" : "Method comparison"}</span></div>
    <div class="comparison-list">${comparisonRows}</div>`;
  $$("[data-benchmark]").forEach((button) => button.addEventListener("click", () => renderBenchmarks(button.dataset.benchmark)));
  $$("[data-benchmark-view]").forEach((button) => button.addEventListener("click", () => renderBenchmarks(item.id, button.dataset.benchmarkView)));
}

function renderRoadmap() {
  $("#roadmap-list").innerHTML = state.data.roadmapList.map((item) => `<article class="roadmap-item"><div class="roadmap-marker">${item.release}</div><h3>${copy(item.title)}</h3><p>${copy(item.text)}</p><span class="roadmap-state">${copy(item.state)}</span></article>`).join("");
}

function renderDemos(active = "all") {
  const groups = state.data.demosGroups;
  const items = active === "all" ? state.data.demosList : state.data.demosList.filter((item) => item.group === active);
  $("#demo-filter").innerHTML = groups.map((group) => {
    const count = group.id === "all" ? state.data.demosList.length : state.data.demosList.filter((item) => item.group === group.id).length;
    return `<button class="demo-filter-button ${group.id === active ? "active" : ""}" type="button" data-demo-group="${group.id}" role="tab">${copy(group.label)} · ${count}</button>`;
  }).join("");
  $("#demo-grid").innerHTML = items.map((item) => `
    <article class="demo-card">
      <video controls preload="none" playsinline poster="${item.poster}">
        <source src="${item.video}" type="video/mp4">
      </video>
      <div><strong>${item.title}</strong><span>${copy(item.subtitle)}</span></div>
    </article>`).join("");
  $$('[data-demo-group]').forEach((button) => button.addEventListener("click", () => renderDemos(button.dataset.demoGroup)));
  $$(".demo-card video").forEach((video) => video.addEventListener("play", () => {
    pauseOtherVideos(video);
  }));
}

function renderDocs() {
  $("#docs-grid").innerHTML = state.data.docsList.map((item) => `<a class="docs-card" href="${copy(item.href)}" target="_blank" rel="noreferrer"><div class="docs-label">${copy(item.label).toUpperCase()}</div><h3>${copy(item.title)}</h3><p>${copy(item.text)}</p><span class="docs-link">${state.lang === "zh" ? "阅读文档" : "Read document"} ↗</span></a>`).join("");
}

function render() {
  renderStaticCopy();
  renderFramework($("[data-framework].active")?.dataset.framework || "data");
  renderModels($("[data-model].active")?.dataset.model || "backbone");
  renderData();
  renderHeroCarousel(state.heroPage);
  renderBenchmarks($("[data-benchmark].active")?.dataset.benchmark || "openfly");
  renderDemos($("[data-demo-group].active")?.dataset.demoGroup || "all");
  renderRoadmap();
  renderDocs();
}

function bindInteractions() {
  $("[data-language]").addEventListener("click", () => {
    state.lang = state.lang === "en" ? "zh" : "en";
    localStorage.setItem("simplenav-lang", state.lang);
    render();
  });
  $$("[data-framework]").forEach((button) => button.addEventListener("click", () => renderFramework(button.dataset.framework)));
  const sections = $$('main section[id]');
  const navLinks = $$('[data-nav]');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) navLinks.forEach((link) => link.classList.toggle("active", link.dataset.nav === entry.target.id));
    });
  }, { rootMargin: "-28% 0px -62% 0px", threshold: 0 });
  sections.forEach((section) => observer.observe(section));
}

async function init() {
  try {
    const response = await fetch("data/site.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`site data request failed: ${response.status}`);
    state.data = await response.json();
    render();
    bindInteractions();
  } catch (error) {
    console.error(error);
    document.body.insertAdjacentHTML("beforeend", `<div class="noscript">Unable to load site data. Open this page through a local web server or GitHub Pages.</div>`);
  }
}

init();
