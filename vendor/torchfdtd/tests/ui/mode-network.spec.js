import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';

async function open(page){
 const p=await (await page.request.get('/api/examples/3d')).json();
 p.materials=[{name:'Guide',model:'dielectric',epsilon_inf:2.25}];
 p.structures=[];
 p.sources=[{id:'mode-source',name:'Mode pulse',enabled:true,kind:'plane',normal:'z',component:'Ex',center:[0,0,-1],size:[1,1,0],pulse:'gaussian',pulse_cycles:3,wavelength:1.55}];
 await page.addInitScript(p=>{localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p));localStorage.removeItem('torchfdtd.activeJob');},p);
 await page.goto('/');await page.getByRole('button',{name:'Mode ports',exact:true}).click();
 const dialog=page.locator('.mode-network-dialog');await expect(dialog).toBeVisible();return dialog;
}
async function json(route,data){await route.fulfill({json:data});}
async function contents(download){return readFile(await download.path(),'utf8');}
const result={status:'completed',channels:[['left',0],['right',0]],s_real:[[.1,.9],[.9,.1]],s_imag:[[.02,.1],[.1,.02]],phase_planes_um:[-.5,.5],objective:.82,material_gradients:{Guide:-.125},mode_summaries:[],admission:{},request_digest:'test-digest'};

test('mode editor edits named ports, imports a separate envelope and exports backend Python',async({page})=>{
 let envelope;
 await page.route('**/api/mode-networks/validate',route=>{envelope=route.request().postDataJSON();return json(route,{summary:'Metadata accepted'});});
 await page.route('**/api/mode-networks/python',route=>route.fulfill({contentType:'text/plain',body:'# Backend-generated fixed mode setup\nprint("mode network")\n'}));
 const d=await open(page);
 await d.getByLabel('Left port name',{exact:true}).fill('input');await d.getByLabel('Left port name',{exact:true}).blur();
 await d.getByLabel('Number of solved modes').fill('2');await d.getByLabel('Right mode indices').fill('0, 1');await d.getByLabel('Right mode indices').blur();
 await d.getByLabel('S quantity').selectOption('imag');await d.getByLabel('Output channel').selectOption(JSON.stringify(['right',1]));
 await d.locator('[data-material="Guide"]').check();
 await d.getByRole('button',{name:'Validate setup',exact:true}).click();await expect(d.getByRole('status')).toContainText('Metadata accepted');
 expect(envelope.version).toBe(1);expect(envelope.normal).toBe('z');expect(envelope.ports[0].name).toBe('input');expect(envelope.ports[1].mode_indices).toEqual([0,1]);expect(envelope.objective).toEqual({input_channel:['input',0],output_channel:['right',1],quantity:'imag'});expect(envelope.differentiate_materials).toEqual(['Guide']);
 const exported=page.waitForEvent('download');await d.getByRole('button',{name:'Export setup JSON'}).click();const saved=JSON.parse(await contents(await exported));expect(saved).toEqual(envelope);
 saved.ports[0].coordinate_um=-.7;
 await d.locator('[data-file]').setInputFiles({name:'setup.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});
 await expect(d.getByRole('status')).toContainText('Setup imported');await expect(d.getByLabel('Left phase plane (µm)')).toHaveValue('-0.7');
 const python=page.waitForEvent('download');await d.getByRole('button',{name:'Export Python'}).click();expect(await contents(await python)).toContain('Backend-generated');
 await d.locator('[data-file]').setInputFiles({name:'project.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved.project))});await expect(d.getByRole('status')).toContainText('version 1');await expect(d.getByLabel('Left port name',{exact:true})).toHaveValue('input');
});

test('typing invalidates delayed validation and prevents stale results or automatic resubmission',async({page})=>{
 let release,start,posts=0;
 const held=new Promise(resolve=>release=resolve),started=new Promise(resolve=>start=resolve);
 await page.route('**/api/mode-networks/validate',async route=>{start();await held;await json(route,{summary:'OLD VALIDATION'});});
 const d=await open(page);await d.getByRole('button',{name:'Validate setup',exact:true}).click();await started;
 await d.getByLabel('Time steps').fill('123');release();await expect(d.getByRole('status')).toContainText('Setup changed');await expect(d.getByRole('status')).not.toContainText('OLD VALIDATION');
 await page.unroute('**/api/mode-networks/validate');await page.route('**/api/mode-networks/validate',route=>json(route,{}));
 let finish,pollStarted;const completion=new Promise(resolve=>finish=resolve),polling=new Promise(resolve=>pollStarted=resolve);
 await page.route('**/api/mode-network-jobs',route=>{posts++;return json(route,{id:'old',status:'queued'});});
 await page.route('**/api/mode-network-jobs/old',async route=>{pollStarted();await completion;await json(route,{id:'old',status:'completed',result});});
 await d.getByRole('button',{name:'Run mode network'}).click();await expect(d.getByRole('button',{name:'Cancel run'})).toBeEnabled();
 await polling;await d.getByLabel('Pulse cycles').fill('4');finish();await expect(d.getByRole('status')).toContainText('Earlier run finished');await expect(d.locator('[data-results]')).toBeEmpty();expect(posts).toBe(1);
 await d.getByLabel('Left mode indices').fill('bad');await expect(d.getByRole('button',{name:'Run mode network'})).toBeDisabled();await expect(d.getByRole('button',{name:'Export setup JSON'})).toBeDisabled();
});

test('completed job shows complex S, material derivatives and result downloads; close cancels its worker',async({page})=>{
 await page.route('**/api/mode-networks/validate',route=>json(route,{}));let sequence=0,cancelled=[];
 await page.route('**/api/mode-network-jobs',route=>json(route,{id:String(++sequence),status:'queued'}));
 await page.route('**/api/mode-network-jobs/*',route=>json(route,sequence===1?{id:'1',status:'completed',result}:{id:'2',status:'running',progress:{phase:'preparing_modes'}}));
 await page.route('**/api/mode-network-jobs/*/cancel',route=>{cancelled.push(route.request().url());return json(route,{status:'cancelling'});});
 const d=await open(page);await d.getByRole('button',{name:'Run mode network'}).click();await expect(d.getByRole('heading',{name:'Complex S matrix'})).toBeVisible();await expect(d.locator('[data-results]')).toContainText('-0.125000');
 await expect(d.getByRole('link',{name:'Download NPZ'})).toHaveAttribute('href','/api/mode-network-jobs/1/download');await expect(d.getByRole('link',{name:'Download S CSV'})).toHaveAttribute('href','/api/mode-network-jobs/1/s.csv');
 await d.getByRole('button',{name:'Run mode network'}).click();await expect(d.getByRole('status')).toContainText('Preparing port modes');await d.getByRole('button',{name:'Close',exact:true}).click();await expect.poll(()=>cancelled.length).toBe(1);expect(cancelled[0]).toContain('/2/cancel');
});

test('edit during submission cancels accepted stale worker and does not display it',async({page})=>{
 await page.route('**/api/mode-networks/validate',route=>json(route,{}));let start,release,cancelled=false;
 const started=new Promise(resolve=>start=resolve),held=new Promise(resolve=>release=resolve);
 await page.route('**/api/mode-network-jobs',async route=>{start();await held;await json(route,{id:'stale',status:'queued'});});
 await page.route('**/api/mode-network-jobs/stale/cancel',route=>{cancelled=true;return json(route,{});});
 const d=await open(page);await d.getByRole('button',{name:'Run mode network'}).click();await started;await d.getByLabel('Time steps').fill('321');release();await expect.poll(()=>cancelled).toBe(true);await expect(d.locator('[data-results]')).toBeEmpty();await expect(d.getByRole('button',{name:'Run mode network'})).toBeEnabled();
});


test('actual CPU imported open guide produces S, material VJP and downloadable evidence',async({page})=>{
 test.setTimeout(240000);
 const envelope=JSON.parse(await readFile(new URL('../../examples/open_mode_network.json',import.meta.url),'utf8'));
 const d=await open(page);
 await d.locator('[data-file]').setInputFiles({name:'open_mode_network.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(envelope))});
 await expect(d.getByRole('status')).toContainText('Setup imported');
 await d.getByRole('button',{name:'Validate setup',exact:true}).click();await expect(d.getByRole('status')).toContainText('Setup validated',{timeout:30000});
 await d.getByRole('button',{name:'Run mode network'}).click();
 await expect(d.getByRole('heading',{name:'Complex S matrix'})).toBeVisible({timeout:180000});
 await expect(d.locator('[data-results]')).toContainText('Design');
 const text=await d.locator('[data-results]').innerText();expect(text).not.toMatch(/NaN|Infinity/);
 await expect(d.locator('[data-request-digest]')).toHaveText(/^[a-f0-9]{64}$/);
 const gradient=Number(await d.locator('[data-results] table').last().locator('tbody td').last().innerText());expect(Number.isFinite(gradient)).toBe(true);expect(Math.abs(gradient)).toBeGreaterThan(0);
 await d.getByRole('heading',{name:'Complex S matrix'}).scrollIntoViewIfNeeded();
 await page.screenshot({path:'results/ui-mode-network.png',fullPage:true});
 const npz=page.waitForEvent('download');await d.getByRole('link',{name:'Download NPZ'}).click();expect((await npz).suggestedFilename()).toMatch(/\.npz$/);
 const csv=page.waitForEvent('download');await d.getByRole('link',{name:'Download S CSV'}).click();expect(await contents(await csv)).toContain('right');
 const python=page.waitForEvent('download');await d.getByRole('button',{name:'Export Python'}).click();expect(await contents(await python)).toContain('run_mode_network');
});
