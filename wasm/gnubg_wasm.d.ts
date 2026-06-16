/* tslint:disable */
/* eslint-disable */

/**
 * Analyze all 21 possible dice rolls for a position.
 *
 * `position_id` — standard backgammon position ID.
 * `depth` — search depth (0 = static).
 */
export function analyze_position(position_id: string, depth: number): any;

/**
 * Find the best move for a given position and dice roll.
 *
 * `position_id` — standard backgammon position ID.
 * `dice` — two digits e.g. "31" or "64" (order doesn't matter).
 * `depth` — search depth (0 = static evaluation only).
 */
export function best_move(position_id: string, dice: string, depth: number): any;

/**
 * Return a short version string for the engine.
 */
export function engine_info(): string;

/**
 * Evaluate a position and return JSON with cubeless and cubeful equity.
 *
 * `position_id` — a standard backgammon position ID (e.g. "4HPwATDgc/ABMA").
 * `match_score` — optional, colon-separated match score e.g. "3:5" (player:opponent away).
 * `cube_value` — optional cube value (defaults to 1).
 */
export function evaluate_position(position_id: string, match_score?: string | null, cube_value?: number | null): any;

/**
 * Initialize the neural-network weights. Must be called once before any other
 * engine function. Returns `true` on success.
 */
export function init_engine(): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly analyze_position: (a: number, b: number, c: number) => any;
    readonly best_move: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly engine_info: () => [number, number];
    readonly evaluate_position: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly init_engine: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
