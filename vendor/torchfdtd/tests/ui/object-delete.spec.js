import {test,expect} from '@playwright/test';

// Per-object deletion from the property panel and the tree rows: the same
// undoable action as the ribbon tool, so undo restores the object.
async function open(page){
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await expect(page.locator('#connection')).not.toContainText('Connecting');
 await expect(page.locator('#mesh-summary strong')).toContainText('×');
 return errors;
}
async function saved(page){return JSON.parse(await page.evaluate(()=>localStorage.getItem('torchfdtd.project.v1')));}

test('delete a structure from its panel, undo it, delete from the row button, delete a source and a monitor',async({page})=>{
 const errors=await open(page);
 await page.locator('[data-add="sphere"]').click();
 const name=await page.locator('#properties [data-path="name"]').inputValue();
 expect(name).toMatch(/^sphere_/);
 await expect(page.locator('#tree .tree-row').filter({hasText:name})).toHaveCount(1);
 await expect(page.locator('#object-count')).toHaveText('6');
 expect((await saved(page)).structures.some(s=>s.name===name)).toBe(true);
 // Panel button, next to Duplicate.
 const panel=page.locator('#properties .object-actions');
 await expect(panel.getByRole('button',{name:'Duplicate '+name,exact:true})).toBeVisible();
 await panel.getByRole('button',{name:'Delete '+name,exact:true}).click();
 await expect(page.locator('#tree .tree-row').filter({hasText:name})).toHaveCount(0);
 await expect(page.locator('#object-count')).toHaveText('5');
 await expect(page.locator('#property-type')).toHaveText('solver');
 expect((await saved(page)).structures.some(s=>s.name===name)).toBe(false);
 await expect(page.locator('#messages')).toContainText(`Deleted ${name}. Undo (Ctrl+Z) restores it.`);
 // Undo restores it in the tree and the model.
 await page.locator('.tree-tools [data-action="undo"]').click();
 await expect(page.locator('#tree .tree-row').filter({hasText:name})).toHaveCount(1);
 await expect(page.locator('#object-count')).toHaveText('6');
 expect((await saved(page)).structures.some(s=>s.name===name)).toBe(true);
 // Row button: hover-visible, keyboard reachable, same action.
 const item=page.locator('#tree .tree-item').filter({hasText:name});
 const trash=item.locator('.row-delete');
 await expect(trash).toHaveAttribute('aria-label','Delete '+name);
 await item.hover();
 await expect(trash).toBeVisible();
 await trash.focus();
 await page.keyboard.press('Enter');
 await expect(page.locator('#tree .tree-row').filter({hasText:name})).toHaveCount(0);
 await expect(page.locator('#object-count')).toHaveText('5');
 expect((await saved(page)).structures.some(s=>s.name===name)).toBe(false);
 // A source and a monitor go the same way, from the panel and from the row.
 await page.locator('[data-select="source"]').click();
 await expect(page.locator('#property-type')).toHaveText('plane');
 await page.locator('#properties .object-actions').getByRole('button',{name:'Delete source',exact:true}).click();
 await expect(page.locator('[data-select="source"]')).toHaveCount(0);
 expect((await saved(page)).sources.length).toBe(0);
 await page.locator('#tree .tree-item').filter({hasText:'output'}).locator('.row-delete').click({force:true});
 await expect(page.locator('[data-select="output"]')).toHaveCount(0);
 await expect(page.locator('[data-select="input"]')).toHaveCount(1);
 expect((await saved(page)).monitors.map(m=>m.id)).toEqual(['input']);
 await expect(page.locator('#object-count')).toHaveText('3');
 await expect(page.locator('#mesh-summary')).toContainText('No enabled sources');
 // Two undos bring both back; the ribbon tool still deletes the selection.
 await page.keyboard.press('Control+z');await page.keyboard.press('Control+z');
 await expect(page.locator('[data-select="output"]')).toHaveCount(1);await expect(page.locator('[data-select="source"]')).toHaveCount(1);
 await page.locator('[data-select="input"]').click();
 await page.locator('.ribbon [data-action="delete"]').click();
 await expect(page.locator('[data-select="input"]')).toHaveCount(0);
 await page.screenshot({path:'results/ui-object-delete.png',fullPage:true});
 expect(errors).toEqual([]);
});
