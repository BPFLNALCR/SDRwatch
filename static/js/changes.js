(function(){
  const STORAGE = {
    baseline: 'SDRWATCH_ACTIVE_BASELINE_ID',
    token: 'SDRWATCH_TOKEN',
  };
  const REFRESH_MS = 45000;
  const state = {
    baselineId: null,
    filter: 'ALL',
    config: {},
    timer: null,
    busy: false,
    payload: null,
  };

  function readInitialState(){
    const el = document.getElementById('changes-state');
    if(!el){ return {}; }
    try{
      return JSON.parse(el.textContent || '{}');
    }catch(err){
      console.warn('Failed to parse changes-state payload', err);
      return {};
    }
  }

  function getStoredBaselineId(){
    const value = localStorage.getItem(STORAGE.baseline);
    return value && value.trim() ? value.trim() : null;
  }

  function persistBaselineId(value){
    if(value){
      localStorage.setItem(STORAGE.baseline, value);
    }else{
      localStorage.removeItem(STORAGE.baseline);
    }
  }

  function getToken(){
    const token = localStorage.getItem(STORAGE.token);
    return token && token.trim() ? token.trim() : null;
  }

  function authHeaders(){
    const headers = {};
    const token = getToken();
    if(token){
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  }

  function formatTimeLabel(ts){
    if(!ts){ return '—'; }
    const date = new Date(ts);
    if(Number.isNaN(date.getTime())){
      return ts;
    }
    const now = new Date();
    const sameDay = date.toISOString().slice(0, 10) === now.toISOString().slice(0, 10);
    return sameDay ? `${date.toISOString().slice(11, 19)}Z` : date.toISOString().slice(0, 16).replace('T', ' ');
  }

  function escapeHtml(value){
    if(value === null || value === undefined){ return ''; }
    return String(value).replace(/[&<>"']/g, (ch)=>({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[ch]));
  }

  function formatNumber(value, digits){
    const num = Number(value);
    if(!Number.isFinite(num)){ return null; }
    return num.toFixed(digits);
  }

  function renderPlaceholder(message){
    const feed = document.getElementById('change-feed');
    if(feed){
      feed.innerHTML = `<div class="text-sm text-slate-300 border border-dashed border-white/20 rounded-xl p-4">${escapeHtml(message)}</div>`;
    }
    const meta = document.getElementById('changes-meta');
    if(meta){
      meta.textContent = message;
    }
  }

  function renderEventCard(event){
    const type = escapeHtml(event.type || 'EVENT');
    const timeLabel = escapeHtml(event.time_label || event.time_utc || '—');
    const summary = escapeHtml(event.summary || '—');
    const details = escapeHtml(event.details || '');
    const freq = formatNumber(event.f_center_hz ? event.f_center_hz / 1e6 : null, 3);
    const bandwidth = formatNumber(event.bandwidth_hz ? event.bandwidth_hz / 1e3 : null, 0);
    const confidenceValue = Number.isFinite(Number(event.confidence)) ? Number(event.confidence) : null;
    const deltaValue = Number.isFinite(Number(event.delta_db)) ? Number(event.delta_db) : null;
    const downtimeValue = Number.isFinite(Number(event.downtime_minutes)) ? Number(event.downtime_minutes) : null;
    const confidence = formatNumber(confidenceValue, 2);
    const delta = formatNumber(deltaValue, 1);
    const downtime = formatNumber(downtimeValue, 0);
    const parts = [];
    if(freq !== null){ parts.push(`Center ${freq} MHz`); }
    if(bandwidth !== null){ parts.push(`BW ${bandwidth} kHz`); }
    if(confidence !== null){ parts.push(`Confidence ${confidence}`); }
    if(delta !== null){ parts.push(`Δ ${delta} dB`); }
    if(event.total_windows){ parts.push(`Total windows ${event.total_windows}`); }
    if(downtime !== null){ parts.push(`Quiet ${downtime} min`); }
    const meta = parts.length ? `<div class="flex flex-wrap gap-3 text-[11px] text-slate-400 mt-3">${parts.map((p)=>`<span>${escapeHtml(p)}</span>`).join('')}</div>` : '';
    return `
      <div class="border border-white/10 rounded-xl p-4 bg-white/5" data-event-type="${type}">
        <div class="flex items-center justify-between gap-3 text-xs text-slate-400">
          <div class="chip text-xs uppercase tracking-wide">${type}</div>
          <div>${timeLabel}</div>
        </div>
        <div class="text-sm font-semibold mt-2">${summary}</div>
        <div class="text-xs text-slate-300 mt-1">${details}</div>
        ${meta}
      </div>
    `;
  }

  function updateFilterClasses(){
    const bar = document.getElementById('change-filter-bar');
    if(!bar){ return; }
    const buttons = bar.querySelectorAll('[data-filter]');
    buttons.forEach((btn)=>{
      const key = (btn.getAttribute('data-filter') || 'ALL').toUpperCase();
      const isActive = key === (state.filter || 'ALL');
      btn.classList.toggle('chip-active', isActive);
    });
  }

  function updateMeta(payload){
    const meta = document.getElementById('changes-meta');
    if(!meta){ return; }
    if(!payload){
      meta.textContent = 'Select a baseline to view changes.';
      return;
    }
    const total = Array.isArray(payload.events) ? payload.events.length : 0;
    const windowMinutes = payload.window_minutes || state.config.window_minutes || 0;
    const refreshed = payload.generated_at ? formatTimeLabel(payload.generated_at) : '—';
    meta.textContent = `${total} events · Window ${windowMinutes || '—'} min · Refreshed ${refreshed}`;
  }

  function renderPayload(payload){
    state.payload = payload || null;
    const feed = document.getElementById('change-feed');
    if(!feed){ return; }
    const events = payload && Array.isArray(payload.events) ? payload.events : [];
    if(!events.length){
      feed.innerHTML = '<div class="text-sm text-slate-300 border border-dashed border-white/20 rounded-xl p-4">No change events in the selected window.</div>';
    }else{
      feed.innerHTML = events.map(renderEventCard).join('');
    }
    updateMeta(payload);
    updateFilterClasses();
  }

  function setMetaError(message){
    const meta = document.getElementById('changes-meta');
    if(meta){
      meta.textContent = message;
    }
  }

  async function fetchChanges(){
    if(!state.baselineId){
      throw new Error('No baseline selected');
    }
    const params = new URLSearchParams();
    if(state.filter && state.filter !== 'ALL'){
      params.set('type', state.filter);
    }
    const minutes = state.config.window_minutes;
    if(minutes){
      params.set('minutes', minutes);
    }
    const url = `/api/baseline/${state.baselineId}/changes${params.toString() ? `?${params.toString()}` : ''}`;
    const resp = await fetch(url, { headers: authHeaders() });
    if(!resp.ok){
      const text = await resp.text();
      throw new Error(text || `Request failed (${resp.status})`);
    }
    return resp.json();
  }

  function updateQueryParams(){
    const url = new URL(window.location.href);
    if(state.baselineId){
      url.searchParams.set('baseline_id', state.baselineId);
    }else{
      url.searchParams.delete('baseline_id');
    }
    if(state.filter && state.filter !== 'ALL'){
      url.searchParams.set('type', state.filter);
    }else{
      url.searchParams.delete('type');
    }
    window.history.replaceState({}, '', url);
  }

  function resetTimer(){
    if(state.timer){
      clearInterval(state.timer);
      state.timer = null;
    }
    if(state.baselineId){
      state.timer = setInterval(fetchAndRender, REFRESH_MS);
    }
  }

  async function fetchAndRender(){
    if(state.busy || !state.baselineId){
      return;
    }
    state.busy = true;
    try{
      const payload = await fetchChanges();
      renderPayload(payload);
    }catch(err){
      console.error(err);
      setMetaError(err.message || 'Failed to refresh change feed.');
    }finally{
      state.busy = false;
    }
  }

  function handleBaselineChange(evt){
    const value = evt.target.value || '';
    state.baselineId = value || null;
    persistBaselineId(state.baselineId);
    updateQueryParams();
    if(!state.baselineId){
      renderPlaceholder('Select a baseline to view changes.');
      return;
    }
    renderPlaceholder('Loading change events…');
    fetchAndRender();
    resetTimer();
  }

  function setFilter(filter){
    const normalized = (filter || 'ALL').toUpperCase();
    if(state.filter === normalized){
      return;
    }
    state.filter = normalized;
    updateFilterClasses();
    updateQueryParams();
    fetchAndRender();
  }

  function setupFilterBar(){
    const bar = document.getElementById('change-filter-bar');
    if(!bar){
      return;
    }
    bar.addEventListener('click', (evt)=>{
      const target = evt.target.closest('[data-filter]');
      if(!target){ return; }
      evt.preventDefault();
      setFilter(target.getAttribute('data-filter'));
    });
  }

  function init(){
    const initial = readInitialState();
    state.config = initial.config || {};
    state.filter = (initial.active_filter || 'ALL').toUpperCase();
    state.baselineId = initial.selected_baseline_id || getStoredBaselineId();
    if(initial.payload && Object.keys(initial.payload).length){
      state.payload = initial.payload;
      renderPayload(initial.payload);
    }
    const select = document.getElementById('changes-baseline');
    if(select){
      if(state.baselineId && Array.from(select.options).some((opt)=>opt.value === state.baselineId)){
        select.value = state.baselineId;
      }else if(select.value){
        state.baselineId = select.value;
      }
      select.addEventListener('change', handleBaselineChange);
    }
    setupFilterBar();
    updateFilterClasses();
    if(state.baselineId){
      persistBaselineId(state.baselineId);
      updateQueryParams();
      resetTimer();
      fetchAndRender();
    }else if(!state.payload){
      renderPlaceholder('Select a baseline to view changes.');
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
