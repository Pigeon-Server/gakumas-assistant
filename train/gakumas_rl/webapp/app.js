/* Gakumas RL Exam Viewer – app.js */
(function () {
  'use strict';

  /* ── State ── */
  let report = null;
  let currentStep = 0;
  let $tooltip = null;

  /* ── DOM refs ── */
  const $picker = document.getElementById('picker');
  const $app = document.getElementById('app');
  const $dropZone = document.getElementById('dropZone');
  const $fileInput = document.getElementById('fileInput');
  const $summaryBar = document.getElementById('summaryBar');
  const $navIndicator = document.getElementById('navIndicator');
  const $btnPrev = document.getElementById('btnPrev');
  const $btnNext = document.getElementById('btnNext');
  const $phoneFrame = document.getElementById('phoneFrame');

  /* ── Resource label map ── */
  const RES_LABELS = {
    block: '元気', review: '好印象', aggressive: '強気',
    concentration: '集中', full_power_point: '全力pt',
    parameter_buff: '好調', lesson_buff: 'レッスン',
    preservation: '温存', over_preservation: '余温存',
    enthusiastic: '熱意', sleepy: '眠気', panic: 'パニック',
  };

  /* ── Helpers ── */
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function signClass(v) { return v > 0 ? 'pos' : v < 0 ? 'neg' : ''; }
  function signStr(v, decimals) {
    const s = v.toFixed(decimals);
    return v > 0 ? '+' + s : s;
  }

  function cardImgSrc(card) {
    if (!card.card_id) return null;
    const upgraded = card.card_id + '_' + (card.upgrade_count || 0) + '.png';
    const fallback = card.card_id + '_0.png';
    return { upgraded, fallback };
  }

  function cardTheme(card, kind) {
    if (kind === 'drink') return 'theme-drink';
    const cat = (card.category || '').toLowerCase();
    if (cat.includes('mental')) return 'theme-mental';
    return 'theme-active';
  }

  function selectedItem(before, action) {
    if (action.slot_group === 'hand' && action.slot_index < before.hand_cards.length)
      return before.hand_cards[action.slot_index];
    if (action.slot_group === 'drink' && action.slot_index < before.drinks.length)
      return before.drinks[action.slot_index];
    return null;
  }

  /* ── Card hover tooltip ── */
  function showCardTooltip(e, info) {
    if (!$tooltip) {
      $tooltip = document.createElement('div');
      $tooltip.className = 'card-tooltip';
      document.body.appendChild($tooltip);
    }

    let html = '';

    if (info.label) {
      html += '<div class="tooltip-title">' + esc(info.label) + '</div>';
    }

    if (info.original_name && info.original_name !== info.label) {
      html += '<div class="tooltip-original">' + esc(info.original_name) + '</div>';
    }

    if (info.description_lines && info.description_lines.length) {
      html += '<div class="tooltip-section">';
      html += '<div class="tooltip-section-title">效果翻译</div>';
      html += '<ul class="tooltip-desc">';
      for (const line of info.description_lines) {
        html += '<li>' + esc(line) + '</li>';
      }
      html += '</ul></div>';
    }

    if (info.original_description_lines && info.original_description_lines.length &&
        JSON.stringify(info.original_description_lines) !== JSON.stringify(info.description_lines)) {
      html += '<div class="tooltip-section">';
      html += '<div class="tooltip-section-title">效果原文</div>';
      html += '<ul class="tooltip-desc">';
      for (const line of info.original_description_lines) {
        html += '<li>' + esc(line) + '</li>';
      }
      html += '</ul></div>';
    }

    $tooltip.innerHTML = html;
    $tooltip.style.display = 'block';

    const x = e.clientX + 15;
    const y = e.clientY + 15;
    const rect = $tooltip.getBoundingClientRect();

    const finalX = (x + rect.width > window.innerWidth) ? e.clientX - rect.width - 15 : x;
    const finalY = (y + rect.height > window.innerHeight) ? e.clientY - rect.height - 15 : y;

    $tooltip.style.left = finalX + 'px';
    $tooltip.style.top = finalY + 'px';
  }

  function hideCardTooltip() {
    if ($tooltip) {
      $tooltip.style.display = 'none';
    }
  }

  function attachCardHoverListeners() {
    document.querySelectorAll('.card-tile').forEach(tile => {
      if (!tile._cardInfo) return;

      tile.addEventListener('mouseenter', (e) => {
        showCardTooltip(e, tile._cardInfo);
      });
      tile.addEventListener('mousemove', (e) => {
        if ($tooltip && $tooltip.style.display === 'block') {
          const x = e.clientX + 15;
          const y = e.clientY + 15;
          const rect = $tooltip.getBoundingClientRect();
          const finalX = (x + rect.width > window.innerWidth) ? e.clientX - rect.width - 15 : x;
          const finalY = (y + rect.height > window.innerHeight) ? e.clientY - rect.height - 15 : y;
          $tooltip.style.left = finalX + 'px';
          $tooltip.style.top = finalY + 'px';
        }
      });
      tile.addEventListener('mouseleave', hideCardTooltip);
    });
  }

  /* ── File loading ── */
  $dropZone.addEventListener('click', () => $fileInput.click());
  $fileInput.addEventListener('change', (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); });
  $dropZone.addEventListener('dragover', (e) => { e.preventDefault(); $dropZone.classList.add('drag-over'); });
  $dropZone.addEventListener('dragleave', () => $dropZone.classList.remove('drag-over'));
  $dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    $dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });

  function loadFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        report = JSON.parse(e.target.result);
        if (!report.trace || !report.trace.length) throw new Error('trace is empty');
        currentStep = 0;
        showApp();
      } catch (err) {
        alert('JSON 解析失败: ' + err.message);
      }
    };
    reader.readAsText(file);
  }

  /* ── Show app ── */
  function showApp() {
    $picker.style.display = 'none';
    $app.classList.add('active');
    renderSummary();
    renderFrame();
  }

  /* ── Summary ── */
  function renderSummary() {
    const s = report.summary;
    const items = [
      ['Scenario', s.scenario],
      ['Backend', s.backend],
      ['Score', s.final_score.toFixed(1)],
      ['Reward', s.total_reward.toFixed(3)],
    ];
    $summaryBar.innerHTML = items.map(([label, value]) =>
      '<div><div class="s-label">' + esc(label) + '</div><div class="s-value">' + esc(String(value)) + '</div></div>'
    ).join('');
  }

  /* ── Navigation ── */
  $btnPrev.addEventListener('click', () => go(-1));
  $btnNext.addEventListener('click', () => go(1));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
  });

  function go(delta) {
    if (!report) return;
    const next = currentStep + delta;
    if (next < 0 || next >= report.trace.length) return;
    currentStep = next;
    renderFrame();
  }

  /* ── Render current frame ── */
  function renderFrame() {
    const frame = report.trace[currentStep];
    const before = frame.before;
    const after = frame.after;
    const action = frame.action;
    const total = report.trace.length;

    $navIndicator.textContent = 'Turn ' + before.turn + ' (' + (currentStep + 1) + '/' + total + ')';
    $btnPrev.disabled = currentStep === 0;
    $btnNext.disabled = currentStep === total - 1;

    let html = '';

    /* HUD */
    const remainTurns = Math.max(before.max_turns - before.turn + 1, 0);
    html += '<div class="hud">';
    html += '<div class="hud-turn-box"><div class="hud-turn-label">残りターン</div><div class="hud-turn-value">' + remainTurns + '</div></div>';
    html += '<div class="hud-score-box"><div class="hud-score-main">' + before.score.toFixed(0) + '</div><div class="hud-score-target">/ ' + before.target_score.toFixed(0) + '</div></div>';
    html += '<div class="hud-right">';
    html += '<div class="hud-stamina"><span class="heart">♥</span> <b>' + before.stamina.toFixed(0) + '</b> / ' + before.max_stamina.toFixed(0) + '</div>';
    html += '<div>姿態 <b>' + esc(before.stance) + '</b></div>';
    html += '<div>出牌 <b>' + before.play_limit + '</b></div>';
    html += '</div></div>';

    /* Resources */
    const resEntries = Object.entries(before.resources).filter(([, v]) => Math.abs(v) > 0.001);
    if (resEntries.length) {
      html += '<div class="resources">';
      for (const [key, val] of resEntries) {
        const label = RES_LABELS[key] || key;
        html += '<span class="res-badge">' + esc(label) + ' <span class="res-val">' + val.toFixed(0) + '</span></span>';
      }
      html += '</div>';
    }

    /* Decision bar */
    const sel = selectedItem(before, action);
    const actionTitle = sel ? (sel.label || action.label) : action.label;
    const scoreDelta = after.score - before.score;
    const staminaDelta = after.stamina - before.stamina;
    html += '<div class="decision-bar">';
    html += '<div><div class="decision-label">モデル選択</div><div class="decision-value">' + esc(actionTitle) + '</div></div>';
    html += '<div class="decision-delta">';
    html += 'reward <span class="' + signClass(frame.reward) + '">' + signStr(frame.reward, 3) + '</span><br>';
    html += 'score <span class="' + signClass(scoreDelta) + '">' + signStr(scoreDelta, 1) + '</span>';
    html += ' · stamina <span class="' + signClass(staminaDelta) + '">' + signStr(staminaDelta, 1) + '</span>';
    html += '</div></div>';

    /* Fallback */
    if (frame.fallback_reason) {
      html += '<div class="fallback-note">' + esc(frame.fallback_reason) + ' → ' + esc(action.label) + '</div>';
    }

    /* Detail box */
    html += renderDetailBox(before, action);

    /* Gimmicks */
    if (before.gimmicks && before.gimmicks.length) {
      html += '<div class="gimmick-row">';
      for (const g of before.gimmicks) {
        const text = (g.text_lines && g.text_lines[0]) || g.effect_type || g.effect_id || '';
        html += '<span class="gimmick-chip"><span class="g-turn">T' + g.start_turn + '</span>' + esc(text) + '</span>';
      }
      html += '</div>';
    }

    /* Hand cards */
    html += '<div class="hand-section">';
    html += '<div class="hand-section-title">手牌</div>';
    html += '<div class="hand-strip">';
    const handCardInfos = [];
    for (const card of before.hand_cards) {
      const isSelected = action.slot_group === 'hand' && action.slot_index === card.slot;
      html += renderCardTile(card, 'hand', isSelected);
      handCardInfos.push(card);
    }
    html += '</div></div>';

    /* Drinks */
    if (before.drinks && before.drinks.length) {
      html += '<div class="drink-section">';
      html += '<div class="hand-section-title">Pドリンク</div>';
      html += '<div class="drink-strip">';
      for (const drink of before.drinks) {
        const isSelected = action.slot_group === 'drink' && action.slot_index === drink.slot;
        const cls = 'drink-tile' + (drink.consumed ? ' consumed' : '') + (isSelected ? ' selected' : '');
        const desc = (drink.preview_lines || []).slice(0, 2).join(' / ');
        html += '<div class="' + cls + '">';
        html += '<div class="drink-name">' + esc(drink.label || '') + '</div>';
        if (desc) html += '<div class="drink-desc">' + esc(desc) + '</div>';
        html += '</div>';
      }
      html += '</div></div>';
    }

    /* Bottom stats */
    const ps = before.parameter_stats;
    const z = before.zones;
    html += '<div class="bottom-stats">';
    html += '<div><div class="stat-group-title">パラメータ</div>';
    html += 'Vo <b>' + ps.vocal.toFixed(0) + '</b> Da <b>' + ps.dance.toFixed(0) + '</b> Vi <b>' + ps.visual.toFixed(0) + '</b></div>';
    html += '<div><div class="stat-group-title">カード</div>';
    html += '山 <b>' + z.deck + '</b> 手 <b>' + z.hand + '</b> 捨 <b>' + z.grave + '</b> 除 <b>' + z.lost + '</b></div>';
    html += '</div>';

    $phoneFrame.innerHTML = html;

    // Attach card info to DOM elements after rendering
    const cardTiles = $phoneFrame.querySelectorAll('.hand-strip .card-tile');
    cardTiles.forEach((tile, idx) => {
      if (idx < handCardInfos.length) {
        tile._cardInfo = handCardInfos[idx];
      }
    });

    attachCardHoverListeners();
  }

  /* ── Detail box ── */
  function renderDetailBox(before, action) {
    const item = selectedItem(before, action);
    if (!item) {
      return '<div class="detail-box system-action">' +
        '<div class="detail-title">' + esc(action.label || 'end_turn') + '</div>' +
        '<div class="detail-sub">システムアクション</div>' +
        '</div>';
    }
    let h = '<div class="detail-box">';
    h += '<div class="detail-title">' + esc(item.label || action.label || '') + '</div>';
    const sub = item.category || item.rarity || action.kind || '';
    if (sub) h += '<div class="detail-sub">' + esc(sub) + '</div>';

    const badges = item.cost_badges || [];
    if (badges.length) {
      h += '<div class="detail-meta">';
      for (const b of badges) {
        const cls = b.startsWith('体力') ? 'meta-pill cost-stamina' : 'meta-pill';
        h += '<span class="' + cls + '">' + esc(b) + '</span>';
      }
      h += '</div>';
    }

    const lines = item.description_lines || item.preview_lines || [];
    if (lines.length) {
      h += '<ul class="detail-desc">';
      for (const line of lines) {
        h += '<li>' + esc(line) + '</li>';
      }
      h += '</ul>';
    }
    h += '</div>';
    return h;
  }

  /* ── Card tile ── */
  function renderCardTile(card, kind, isSelected) {
    const theme = cardTheme(card, kind);
    let cls = 'card-tile ' + theme;
    if (isSelected) cls += ' selected';
    if (!card.available) cls += ' disabled';

    let h = '<div class="' + cls + '">';

    const stamina = card.stamina || 0;
    if (stamina) {
      const costCls = 'card-cost cost-stamina';
      h += '<div class="' + costCls + '">' + Math.abs(stamina).toFixed(0) + '</div>';
    }

    const imgs = cardImgSrc(card);
    h += '<div class="card-img-wrap">';
    if (imgs) {
      h += '<img src="skill_card/' + esc(imgs.upgraded) + '" alt="' + esc(card.label || '') + '" loading="lazy" onerror="this.onerror=null; this.src=\'skill_card/' + esc(imgs.fallback) + '\'; this.onerror=function(){this.parentNode.innerHTML=\'<div class=no-img>NO IMG</div>\';}">';
    } else {
      h += '<div class="no-img">NO IMG</div>';
    }
    h += '</div>';

    h += '<div class="card-name-label">' + esc(card.label || '') + '</div>';
    const catTag = card.category || '';
    if (catTag) h += '<div class="card-category-tag">' + esc(catTag) + '</div>';

    if (isSelected) h += '<div class="select-tag">SELECT</div>';

    h += '</div>';
    return h;
  }

})();
