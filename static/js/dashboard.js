(function(){
  const STORAGE_KEY = 'SDRWATCH_ACTIVE_BASELINE_ID';

  function persistBaseline(id){
    if(id && String(id).trim()){
      localStorage.setItem(STORAGE_KEY, String(id).trim());
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  function submitForm(form){
    if(!form){ return; }
    if(typeof form.requestSubmit === 'function'){
      form.requestSubmit();
    } else {
      form.submit();
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    const state = window.SDRWATCH_DASHBOARD_STATE || {};
    const select = document.querySelector('[data-dashboard-baseline-select]');
    const form = document.querySelector('[data-dashboard-baseline-form]');
    const stored = localStorage.getItem(STORAGE_KEY);

    if(state.selected_baseline_id){
      persistBaseline(state.selected_baseline_id);
    } else if(stored && select && !select.value){
      select.value = stored;
    }

    if(select){
      select.addEventListener('change', ()=>{
        const value = select.value || '';
        persistBaseline(value);
        submitForm(form || select.closest('form'));
      });
    }

    const refreshBtn = document.querySelector('[data-dashboard-refresh]');
    if(refreshBtn){
      refreshBtn.addEventListener('click', (evt)=>{
        evt.preventDefault();
        submitForm(form || refreshBtn.closest('form'));
      });
    }

    document.querySelectorAll('[data-change-filter-link]').forEach((link)=>{
      link.addEventListener('click', ()=>{
        const current = select && select.value ? select.value : state.selected_baseline_id;
        if(current){ persistBaseline(current); }
      });
    });
  });
})();
