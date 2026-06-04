/**
 * Helper kecil yang dipakai di banyak komponen.
 */

/** Gabungkan className bersyarat menjadi satu string (mirip clsx versi ringan). */
export function cn(...classes: string[]): string {
  return classes.filter(Boolean).join(" ");
}
