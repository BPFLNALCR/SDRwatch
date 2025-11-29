(function(){
  const STORAGE_KEYS = {
    baseline: 'SDRWATCH_ACTIVE_BASELINE_ID',
    token: 'SDRWATCH_TOKEN',
  };
  const state = {
    currentBaselineId: null,
    lastRequestId: 0,
    elements: null,
  };

  function getConfig(){
    return window.SDRWATCH_DASHBOARD_CONFIG || {};
  }

  function getBaselineId(){
    const value = localStorage.getItem(STORAGE_KEYS.baseline);
    return value && value.trim() ? value.trim() : null;
  }

  function authHeaders(){
    const headers = {};
    const token = localStorage.getItem(STORAGE_KEYS.token);
    if(token){
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  }

  async function fetchJson(url){
    const resp = await fetch(url, { headers: authHeaders() });
    if(!resp.ok){
      const text = await resp.text();
      throw new Error(text || `Request failed (${resp.status})`);
    }
    return resp.json();
  }

  function setText(el, text){
    if(el){ el.textContent = text; }
  }

  function setBodyMessage(el, text, cls = 'text-sm muted'){
    if(el){
      const safeText = escapeHtml(text);
      el.innerHTML = `<div class="${cls}">${safeText}</div>`;
    }
  }

  function formatMHz(value){
    if(value === null || value === undefined || Number.isNaN(Number(value))){
      return '—';
    }
    return (Number(value) / 1e6).toFixed(3);
  }

  function formatKHz(value){
    if(value === null || value === undefined || Number.isNaN(Number(value))){
      return '—';
    }
    return (Number(value) / 1e3).toFixed(1);
  }

  function formatTimeHHMMSS(ts){
    if(!ts){ return '—'; }
    const date = new Date(ts);
    if(Number.isNaN(date.getTime())){
      return ts;
    }
    return date.toISOString().slice(11, 19) + 'Z';
  }

  function formatShortDate(ts){
    if(!ts){ return '—'; }
    const date = new Date(ts);
    if(Number.isNaN(date.getTime())){
      return ts;
    }
    return date.toISOString().slice(0, 16).replace('T', ' ');
  }

  function formatConfidence(value){
    if(value === null || value === undefined || Number.isNaN(Number(value))){
      return '—';
    }
    return `${Math.round(Number(value) * 100)}%`;
  }

  function escapeHtml(value){
    if(value === undefined || value === null){ return ''; }
    return String(value).replace(/[&<>"']/g, (ch)=>({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[ch]));
  }

  function escapeAttr(value){
    return escapeHtml(value);
  }

  function collectElements(){
    const snapshotBody = document.getElementById('tactical-snapshot-body');
    if(!snapshotBody){ return null; }
    return {
      snapshotBody,
      snapshotLabel: document.getElementById('tactical-baseline-label'),
      refreshBtn: document.getElementById('tactical-refresh-btn'),
      activeBody: document.getElementById('active-signals-body'),
      activeMeta: document.getElementById('active-signals-meta'),
      hotspotStrip: document.getElementById('hotspot-strip'),
      hotspotMeta: document.getElementById('hotspot-meta'),
      hotspotAxis: document.getElementById('hotspot-axis'),
      hotspotEmpty: document.getElementById('hotspot-empty'),
    };
  }

  function renderBaselineMissing(elements){
    setText(elements.snapshotLabel, 'No baseline selected.');
    setBodyMessage(elements.snapshotBody, 'Select a baseline on the Control page to unlock tactical data.');
    setBodyMessage(elements.activeBody, 'Once scans run against a baseline, recent activity will appear here.');
    setText(elements.activeMeta, 'Window —');
    if(elements.hotspotStrip){ elements.hotspotStrip.innerHTML = ''; }
    if(elements.hotspotAxis){ elements.hotspotAxis.innerHTML = ''; }
    setText(elements.hotspotMeta, '—');
    setBodyMessage(elements.hotspotEmpty, 'Select a baseline to render occupancy.');
  }

  function renderSnapshot(elements, payload){
    if(!payload || !payload.baseline){
      setBodyMessage(elements.snapshotBody, 'Baseline not found.');
      return;
    }
    const baseline = payload.baseline;
    const snapshot = payload.snapshot || {};
    const freqLabel = `${formatMHz(baseline.freq_start_hz)}–${formatMHz(baseline.freq_stop_hz)} MHz`;
    const lines = [
      `<div><span class="muted">Range:</span> ${freqLabel}</div>`,
      `<div><span class="muted">Created:</span> ${escapeHtml(baseline.created_at || '—')}</div>`,
    ];
    if(baseline.location_lat !== null && baseline.location_lat !== undefined && baseline.location_lon !== null && baseline.location_lon !== undefined){
      lines.push(`<div><span class="muted">Location:</span> ${(Number(baseline.location_lat)).toFixed(4)}, ${(Number(baseline.location_lon)).toFixed(4)}</div>`);
    }
    if(baseline.antenna){
      lines.push(`<div><span class="muted">Antenna:</span> ${escapeHtml(baseline.antenna)}</div>`);
    }
    const metrics = [
      { label: 'Persistent signals', value: snapshot.persistent_signals ?? '—' },
      { label: 'Last update', value: snapshot.last_update ? formatShortDate(snapshot.last_update) : '—' },
      { label: 'Recent new', value: `${snapshot.recent_new ?? 0} / ${snapshot.recent_window_minutes || 0} min` },
    ];
    const metricsHtml = metrics.map((m)=>`
      <div class="text-center">
        <div class="text-[11px] uppercase muted">${m.label}</div>
        <div class="text-xl font-semibold">${m.value}</div>
      </div>
    `).join('');
    elements.snapshotBody.innerHTML = `
      <div class="text-sm space-y-1">${lines.join('')}</div>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">${metricsHtml}</div>
    `;
    setText(elements.snapshotLabel, `#${baseline.id} · ${baseline.name}`);
  }

  function renderActiveSignals(elements, payload){
    const snapshot = payload?.snapshot || {};
    const activeSignals = payload?.active_signals || [];
    const windowMinutes = snapshot.active_window_minutes || getConfig().active_window_minutes;
    setText(elements.activeMeta, windowMinutes ? `Window ${windowMinutes} min` : 'Window —');
    if(!activeSignals.length){
      setBodyMessage(elements.activeBody, 'No signals observed in this window.');
      return;
    }
    const rows = activeSignals.map((sig)=>{
      const freq = formatMHz(sig.f_center_hz);
      const bw = formatKHz(sig.bandwidth_hz);
      const conf = formatConfidence(sig.confidence);
      const lastSeen = formatTimeHHMMSS(sig.last_seen_utc);
      return `
        <tr class="hover:bg-white/5">
          <td class="td">${freq}</td>
          <td class="td">${bw}</td>
          <td class="td">${conf}</td>
          <td class="td">${lastSeen}</td>
        </tr>
      `;
    }).join('');
    elements.activeBody.innerHTML = `
      <div class="overflow-x-auto">
        <table class="table text-xs">
          <thead>
            <tr class="text-slate-400">
              <th class="th">Center (MHz)</th>
              <th class="th">Bandwidth (kHz)</th>
              <th class="th">Confidence</th>
              <th class="th">Last seen (UTC)</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }

  function renderHotspots(elements, payload){
    if(!elements.hotspotStrip){ return; }
    const buckets = payload?.buckets || [];
    if(!buckets.length || !payload?.has_samples){
      elements.hotspotStrip.innerHTML = '';
      if(elements.hotspotAxis){ elements.hotspotAxis.innerHTML = ''; }
      setText(elements.hotspotMeta, '—');
      setBodyMessage(elements.hotspotEmpty, 'No baseline stats available yet.');
      return;
    }
    const maxOcc = payload.max_occ || 0;
    const freqStart = payload.freq_start_hz || 0;
    const freqStop = payload.freq_stop_hz || 0;
    const bucketWidthHz = payload.bucket_width_hz || 0;
    const cells = buckets.map((bucket)=>{
      const occ = bucket.avg_occ || 0;
      const intensity = maxOcc > 0 ? Math.min(1, occ / maxOcc) : 0;
      const isHot = intensity >= 0.7;
      const start = formatMHz(bucket.f_low_hz);
      const stop = formatMHz(bucket.f_high_hz);
      const powerVal = Number(bucket.avg_power_ema);
      const powerText = Number.isFinite(powerVal) ? `${powerVal.toFixed(1)} dB` : '—';
      const title = `${start}–${stop} MHz\nocc ${(occ * 100).toFixed(1)}%\npower ${powerText}`;
      return `<div class="hotspot-cell" data-hot="${isHot}" style="--intensity:${intensity.toFixed(2)}" title="${escapeAttr(title)}"></div>`;
    }).join('');
    elements.hotspotStrip.innerHTML = cells;
    if(elements.hotspotEmpty){ elements.hotspotEmpty.innerHTML = ''; }
    if(elements.hotspotAxis){
      const mid = formatMHz((freqStart + freqStop) / 2);
      elements.hotspotAxis.innerHTML = `<span>${formatMHz(freqStart)} MHz</span><span>${mid} MHz</span><span>${formatMHz(freqStop)} MHz</span>`;
    }
    const bucketWidthMhz = bucketWidthHz ? (bucketWidthHz / 1e6).toFixed(3) : '—';
    setText(elements.hotspotMeta, `Buckets ${buckets.length} · ${bucketWidthMhz} MHz each`);
  }

  function loadTacticalData(elements){
    const baselineId = getBaselineId();
    if(!baselineId){
      state.currentBaselineId = null;
      renderBaselineMissing(elements);
      return;
    }
    state.currentBaselineId = baselineId;
    setBodyMessage(elements.snapshotBody, 'Loading baseline snapshot…', 'text-sm text-slate-200');
    setBodyMessage(elements.activeBody, 'Loading active signals…', 'text-sm text-slate-200');
    setBodyMessage(elements.hotspotEmpty, 'Loading hotspots…', 'text-sm text-slate-200');
    const requestId = ++state.lastRequestId;
    fetchJson(`/api/baseline/${baselineId}/tactical`).then((data)=>{
      if(requestId !== state.lastRequestId){ return; }
      renderSnapshot(elements, data);
      renderActiveSignals(elements, data);
    }).catch((err)=>{
      if(requestId !== state.lastRequestId){ return; }
      setBodyMessage(elements.snapshotBody, `Failed to load snapshot: ${err.message}`, 'text-sm text-red-300');
      setBodyMessage(elements.activeBody, 'Unable to load active signals.', 'text-sm text-red-300');
    });
    fetchJson(`/api/baseline/${baselineId}/hotspots`).then((data)=>{
      if(requestId !== state.lastRequestId){ return; }
      renderHotspots(elements, data);
    }).catch((err)=>{
      if(requestId !== state.lastRequestId){ return; }
      setBodyMessage(elements.hotspotEmpty, `Failed to load hotspots: ${err.message}`, 'text-sm text-red-300');
      if(elements.hotspotStrip){ elements.hotspotStrip.innerHTML = ''; }
      if(elements.hotspotAxis){ elements.hotspotAxis.innerHTML = ''; }
      setText(elements.hotspotMeta, '—');
    });
  }

  function initDashboardPanels(){
    const elements = collectElements();
    if(!elements){ return; }
    state.elements = elements;
    if(elements.refreshBtn && !elements.refreshBtn.dataset.bound){
      elements.refreshBtn.dataset.bound = '1';
      elements.refreshBtn.addEventListener('click', ()=> loadTacticalData(elements));
    }
    loadTacticalData(elements);
  }

  document.addEventListener('DOMContentLoaded', initDashboardPanels);
  document.body.addEventListener('htmx:afterSwap', (evt)=>{
    const target = evt.detail && evt.detail.target;
    if(target && target.id === 'dashboard-shell'){
      initDashboardPanels();
    }
  });
})();
