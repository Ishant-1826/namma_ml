/**
 * minimap.ts
 * Offscreen cached minimap renderer for 2D occupancy grid.
 * Key B toggles, default OFF.
 * +Z points DOWN the image (matching top.png).
 * Occupied cells dark, walkable cells light, free but not walkable cells mid-grey.
 * Overlays must not take pointer events.
 */

import { OccupancyGrid } from "./collision";

export class Minimap {
  readonly container: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  private readonly _ctx: CanvasRenderingContext2D;
  private readonly _offscreenCanvas: HTMLCanvasElement;
  private readonly _offscreenCtx: CanvasRenderingContext2D;
  private _grid: OccupancyGrid | null = null;
  private _visible = false;
  private _lastDrawTime = 0;

  constructor(container: HTMLElement) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.id = "minimap-canvas";
    this.canvas.width = 220;
    this.canvas.height = 220;
    this.container.appendChild(this.canvas);

    this._ctx = this.canvas.getContext("2d")!;
    this._offscreenCanvas = document.createElement("canvas");
    this._offscreenCtx = this._offscreenCanvas.getContext("2d")!;

    this.setVisible(false);
  }

  setGrid(grid: OccupancyGrid): void {
    this._grid = grid;
    this._offscreenCanvas.width = grid.width;
    this._offscreenCanvas.height = grid.height;

    // Render occupancy grid ONCE into offscreen canvas
    const imgData = this._offscreenCtx.createImageData(grid.width, grid.height);
    const pixels = imgData.data;

    for (let r = 0; r < grid.height; r++) {
      for (let c = 0; c < grid.width; c++) {
        const idx = r * grid.width + c;
        const pIdx = idx * 4;
        const isOcc = grid.data[idx] === 1;
        const isWalk = grid.walkable ? grid.walkable[idx] === 1 : !isOcc;

        if (isOcc) {
          // Occupied: dark (#12151b)
          pixels[pIdx] = 18;
          pixels[pIdx + 1] = 21;
          pixels[pIdx + 2] = 27;
          pixels[pIdx + 3] = 235;
        } else if (isWalk) {
          // Walkable: light (#e2e8f0)
          pixels[pIdx] = 226;
          pixels[pIdx + 1] = 232;
          pixels[pIdx + 2] = 240;
          pixels[pIdx + 3] = 245;
        } else {
          // Free but not walkable: mid-grey (#64748b)
          pixels[pIdx] = 100;
          pixels[pIdx + 1] = 116;
          pixels[pIdx + 2] = 139;
          pixels[pIdx + 3] = 230;
        }
      }
    }

    this._offscreenCtx.putImageData(imgData, 0, 0);

    // Aspect-correct canvas size (width ~220px)
    const aspect = grid.height / grid.width;
    this.canvas.width = 220;
    this.canvas.height = Math.round(220 * aspect);
  }

  toggle(): boolean {
    this.setVisible(!this._visible);
    return this._visible;
  }

  setVisible(visible: boolean): void {
    this._visible = visible;
    this.container.classList.toggle("hidden", !visible);
  }

  get isVisible(): boolean {
    return this._visible;
  }

  /**
   * Redraws player position and heading arrow at most 10 times per second (>= 100 ms).
   * +Z points DOWN the canvas.
   */
  update(playerX: number, playerZ: number, yawRad: number, now: number): void {
    if (!this._visible || !this._grid) return;
    if (now - this._lastDrawTime < 100) return;
    this._lastDrawTime = now;

    const ctx = this._ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Blit pre-rendered offscreen grid
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this._offscreenCanvas, 0, 0, w, h);

    // Map player position (world X, Z) to minimap pixels
    const originX = this._grid.origin[0];
    const originZ = this._grid.origin[1];
    const cellSize = this._grid.cell_size;

    const col = (playerX - originX) / cellSize;
    const row = (playerZ - originZ) / cellSize;

    const px = (col / this._grid.width) * w;
    const py = (row / this._grid.height) * h;

    // Draw player dot
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#06b6d4";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    // Draw heading arrow (yaw 0 points North / -Z, which is UP the image: dirZ < 0)
    // +Z is DOWN in canvas coords (py + dirZ * len)
    const dirX = -Math.sin(yawRad);
    const dirY = -Math.cos(yawRad); // dirY < 0 when pointing North (-Z)
    const arrowLen = 12;

    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + dirX * arrowLen, py + dirY * arrowLen);
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Arrowhead tip
    const headLen = 5;
    const angle = Math.atan2(dirY, dirX);
    ctx.beginPath();
    ctx.moveTo(px + dirX * arrowLen, py + dirY * arrowLen);
    ctx.lineTo(
      px + dirX * arrowLen - headLen * Math.cos(angle - Math.PI / 6),
      py + dirY * arrowLen - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      px + dirX * arrowLen - headLen * Math.cos(angle + Math.PI / 6),
      py + dirY * arrowLen - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = "#06b6d4";
    ctx.fill();
  }
}
