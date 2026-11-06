import {test,expect} from '@playwright/test';
import fs from 'node:fs';

// G8-04: editing integrity of the workbench. Each test starts from the 2D
// waveguide example in a fresh browser context (empty browser storage).
const HISTORY_LIMIT=80;

async function open(page){
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await expect(page.locator('#connection')).not.toContainText('Connecting');
 await expect(page.locator('#mesh-summary strong')).toContainText('×');
 return errors;
}
async function setNumber(page,label,value){const input=page.getByLabel(label,{exact:true});await input.fill(value);await input.press('Tab');}
async function saved(page){return JSON.parse(await page.evaluate(()=>localStorage.getItem('torchfdtd.project.v1')));}
async function revision(page){return (await saved(page)).revision||0;}

test('undo and redo restore edits and the history is bounded',async({page})=>{
 test.setTimeout(180000);
 const errors=await open(page);
 await page.locator('[data-select="waveguide"]').click();
 await setNumber(page,'y span','0.8');await setNumber(page,'y span','0.9');
 await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.9');
 await page.locator('.tree-tools [data-action="undo"]').click();
 await page.locator('[data-select="waveguide"]').click();await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.8');
 await page.locator('.tree-tools [data-action="undo"]').click();
 await page.locator('[data-select="waveguide"]').click();await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.65');
 await page.locator('.tree-tools [data-action="redo"]').click();
 await page.locator('[data-select="waveguide"]').click();await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.8');
 await page.keyboard.press('Control+y');
 await page.locator('[data-select="waveguide"]').click();await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.9');
 const before=await revision(page);
 // A bounded history: after HISTORY_LIMIT + 5 edits, at most HISTORY_LIMIT undo steps apply.
 await page.locator('[data-select="fdtd"]').click();
 for(let i=0;i<HISTORY_LIMIT+5;i++)await setNumber(page,'time steps',String(200+i));
 await expect(page.getByLabel('time steps',{exact:true})).toHaveValue(String(200+HISTORY_LIMIT+4));
 expect(await revision(page)).toBe(before+HISTORY_LIMIT+5);
 for(let i=0;i<HISTORY_LIMIT;i++)await page.locator('.tree-tools [data-action="undo"]').click();
 await page.locator('[data-select="fdtd"]').click();await expect(page.getByLabel('time steps',{exact:true})).toHaveValue('204');
 await page.locator('.tree-tools [data-action="undo"]').click();
 await page.locator('[data-select="fdtd"]').click();await expect(page.getByLabel('time steps',{exact:true})).toHaveValue('204');
 expect(await revision(page)).toBeGreaterThan(before+HISTORY_LIMIT+5);
 expect(errors).toEqual([]);
});

test('multi-select, duplicate all, copy, paste and delete all',async({page})=>{
 const errors=await open(page);
 await page.locator('[data-select="waveguide"]').click();
 await page.locator('[data-select="source"]').click({modifiers:['Control']});
 await expect(page.locator('#property-type')).toHaveText('selection');
 await expect(page.locator('#properties')).toContainText('2 objects selected');
 await expect(page.locator('#tree .tree-row.selected')).toHaveCount(2);
 await page.locator('[data-select="source"]').click({modifiers:['Control']});
 await expect(page.locator('#tree .tree-row.selected')).toHaveCount(1);
 await expect(page.locator('#property-type')).toHaveText('rectangle');
 await page.locator('[data-select="source"]').click({modifiers:['Control']});
 await page.locator('.ribbon [data-action="duplicate"]').click();
 await expect(page.locator('#tree')).toContainText('waveguide_copy');await expect(page.locator('#tree')).toContainText('source_copy');
 await expect(page.locator('#object-count')).toHaveText('7');
 await expect(page.locator('#properties')).toContainText('2 objects selected');
 await page.keyboard.press('Control+c');await expect(page.locator('#messages')).toContainText('Copied 2 objects');
 await page.keyboard.press('Control+v');await expect(page.locator('#messages')).toContainText('Pasted 2 objects');
 await expect(page.locator('#tree')).toContainText('waveguide_copy_copy');await expect(page.locator('#tree')).toContainText('source_copy_copy');
 await expect(page.locator('#object-count')).toHaveText('9');
 const project=await saved(page);
 const original=project.structures.find(s=>s.id==='waveguide'),pasted=project.structures.find(s=>s.name==='waveguide_copy_copy');
 expect(pasted.center[0]).toBeCloseTo(original.center[0]+.4,6);expect(pasted.size).toEqual(original.size);expect(pasted.id).not.toBe(original.id);
 await page.locator('#properties').getByRole('button',{name:'Delete all',exact:true}).click();
 await expect(page.locator('#tree')).not.toContainText('_copy_copy');await expect(page.locator('#object-count')).toHaveText('7');
 await expect(page.locator('#messages')).toContainText('Deleted 2 objects');
 await page.locator('.ribbon [data-action="paste"]').click();await expect(page.locator('#object-count')).toHaveText('9');
 expect(errors).toEqual([]);
});

test('autosave to browser storage and recovery after reload',async({page})=>{
 const errors=await open(page);
 await page.locator('[data-select="waveguide"]').click();
 await setNumber(page,'y span','0.8');
 await page.locator('.ribbon [data-action="duplicate"]').click();await expect(page.locator('#tree')).toContainText('waveguide_copy');
 await expect.poll(async()=>(await saved(page)).content_sha256).toMatch(/^[0-9a-f]{64}$/);
 const stored=await saved(page);expect(stored.revision).toBeGreaterThanOrEqual(2);
 await page.reload();await expect(page.locator('#tree')).toContainText('waveguide_copy');
 await expect(page.locator('#messages')).toContainText(`Recovered the autosaved project "${stored.name}" at revision ${stored.revision}`);
 await expect(page.locator('#messages')).not.toContainText('does not match');
 await expect(page.locator('#project-title')).toContainText(`rev ${stored.revision}`);
 await page.locator('[data-select="waveguide"]').click();await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.8');
 await setNumber(page,'y span','0.7');
 expect(await revision(page)).toBe(stored.revision+1);
 expect(errors).toEqual([]);
});

test('the saved project carries a monotonic revision and a content hash the server confirms',async({page})=>{
 const errors=await open(page);
 const downloads=[];page.on('download',d=>downloads.push(d));
 const start=await revision(page);
 await page.locator('[data-select="waveguide"]').click();
 await setNumber(page,'y span','0.8');await setNumber(page,'x span','7');await setNumber(page,'y span','0.75');
 expect(await revision(page)).toBe(start+3);
 await page.keyboard.press('Control+s');
 await expect.poll(()=>downloads.length).toBe(1);
 const project=JSON.parse(fs.readFileSync(await downloads[0].path(),'utf8'));
 expect(project.revision).toBe(start+3);expect(project.content_sha256).toMatch(/^[0-9a-f]{64}$/);
 expect(project.schema_version).toBe(1);
 const check=await (await page.request.post('/api/validate',{data:project})).json();
 expect(check.stored_content_sha256_matches).toBe(true);expect(check.revision).toBe(start+3);
 project.region.steps+=1;
 expect((await (await page.request.post('/api/validate',{data:project})).json()).stored_content_sha256_matches).toBe(false);
 await expect(page.locator('#messages')).toContainText(`Project saved as JSON (revision ${start+3}, content hash ${check.content_sha256.slice(0,12)}`);
 // Undo restores an older snapshot, yet the counter keeps increasing.
 await page.locator('.tree-tools [data-action="undo"]').click();
 expect(await revision(page)).toBe(start+4);
 expect(errors).toEqual([]);
});

test('non-finite and out-of-range values are refused with a visible message',async({page})=>{
 const errors=await open(page);
 await page.locator('[data-select="waveguide"]').click();
 const span=page.getByLabel('y span',{exact:true});
 await span.fill('-1');await span.press('Tab');
 await expect(page.locator('#properties .field-error')).toHaveText('y span must be at least 0.001 µm.');
 await expect(span).toHaveAttribute('aria-invalid','true');
 await expect(page.locator('#toast')).toContainText('y span must be at least 0.001 µm. The previous value is kept.');
 await expect(page.locator('#messages')).toContainText('Rejected input: y span must be at least');
 expect((await saved(page)).structures.find(s=>s.id==='waveguide').size[1]).toBe(0.65);
 await span.fill('');await span.press('Tab');
 await expect(page.locator('#properties .field-error')).toHaveText('y span must be a finite number in µm.');
 await span.fill('1e999');await span.press('Tab');
 await expect(page.locator('#properties .field-error')).toHaveText('y span must be a finite number in µm.');
 expect((await saved(page)).structures.find(s=>s.id==='waveguide').size[1]).toBe(0.65);
 await span.fill('0.8');await span.press('Tab');
 await expect(page.locator('#properties .field-error')).toHaveCount(0);
 await expect(page.getByLabel('y span',{exact:true})).toHaveValue('0.8');
 expect((await saved(page)).structures.find(s=>s.id==='waveguide').size[1]).toBe(0.8);
 await page.locator('[data-select="fdtd"]').click();
 await setNumber(page,'time steps','5');
 await expect(page.locator('#properties .field-error')).toHaveText('time steps must be at least 10.');
 expect((await saved(page)).region.steps).toBe(1000);
 // The server refuses what the browser refused, so a hand-edited file cannot bypass the check.
 const project=await saved(page);project.region.steps=5;
 expect((await page.request.post('/api/validate',{data:project})).status()).toBe(422);
 expect(errors).toEqual([]);
});

test('results are marked stale when the plan changes after a run and current again when it is restored',async({page})=>{
 test.setTimeout(120000);
 const errors=await open(page);
 await page.locator('[data-select="fdtd"]').click();
 await page.getByLabel('resource',{exact:true}).selectOption('cpu');
 await setNumber(page,'time steps','300');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const job=await (await submitted).json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 await expect(page.locator('#results-tree')).toContainText('field snapshots');
 await expect(page.locator('#stale-banner')).toBeHidden();
 await expect(page.locator('#field-label')).not.toContainText('STALE');
 const run=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(run.plan_hash).toMatch(/^[0-9a-f]{64}$/);
 await page.locator('#layout-button').click();
 await expect(page.locator('#results-tree')).toContainText('field snapshots');
 await expect(page.locator('#stale-banner')).toBeHidden();
 await page.locator('[data-select="waveguide"]').click();
 const changed=page.waitForResponse(r=>r.url().endsWith('/api/validate'));
 await setNumber(page,'y span','0.8');
 const current=await (await changed).json();
 expect(current.plan_hash).not.toBe(run.plan_hash);
 await expect(page.locator('#stale-banner')).toBeVisible();
 await expect(page.locator('#stale-banner')).toContainText(`Stale results: the project changed since run revision ${run.revision}`);
 await expect(page.locator('#results-tree')).toHaveClass(/stale/);
 await page.locator('#results-tree button').first().click();
 await expect(page.locator('#field-label')).toContainText('STALE');
 await expect(page.locator('#field-view')).toHaveClass(/stale/);
 await page.locator('[data-tab="geometry"]').click();
 const restored=page.waitForResponse(r=>r.url().endsWith('/api/validate'));
 await page.locator('.tree-tools [data-action="undo"]').click();
 expect((await (await restored).json()).plan_hash).toBe(run.plan_hash);
 await expect(page.locator('#stale-banner')).toBeHidden();
 await expect(page.locator('#results-tree')).not.toHaveClass(/stale/);
 await page.locator('[data-select="fdtd"]').click();
 await setNumber(page,'time steps','320');
 await expect(page.locator('#stale-banner')).toBeVisible();
 await page.locator('[data-select="waveguide"]').click();
 await setNumber(page,'y span','-3');
 await expect(page.locator('#stale-banner')).toBeVisible();
 await expect(page.locator('#properties .field-error')).toBeVisible();
 expect(errors).toEqual([]);
});
