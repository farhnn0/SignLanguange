/**
 * Menampilkan grid 8 video contoh gesture kata BISINDO.
 */

"use client";

import { useRef } from "react";
import { Card } from "./ui";
import { BookOpen } from "lucide-react";

type GestureItem = {
  label: string;
  src: string;
  hint: string;
};

const GESTURES: GestureItem[] = [
  {
    label: "Menulis",
    src: "/samples/Menulis.mp4",
    hint: "Gerakan menulis di telapak tangan",
  },
  {
    label: "Terima Kasih",
    src: "/samples/TerimaKasih.mp4",
    hint: "Tangan ke dagu lalu sapukan ke depan",
  },
  {
    label: "Makan",
    src: "/samples/Makan.mp4",
    hint: "Tangan mengarah ke mulut, berulang",
  },
  {
    label: "Belajar",
    src: "/samples/Belajar.mp4",
    hint: "Seperti membuka buku / membaca",
  },
  {
    label: "Halo",
    src: "/samples/Halo.mp4",
    hint: "Lambaian tangan singkat",
  },
  {
    label: "Sekian",
    src: "/samples/Sekian.mp4",
    hint: "Gerakan tangan menutup / selesai",
  },
  {
    label: "Olahraga",
    src: "/samples/Olahraga.mp4",
    hint: "Kedua tangan menirukan aktivitas fisik",
  },
  {
    label: "Keluarga",
    src: "/samples/Keluarga.mp4",
    hint: "Kedua tangan membentuk lingkaran kelompok",
  },
];

function GestureCard({ item }: { item: GestureItem }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Play video saat pointer masuk
  const handleMouseEnter = () => {
    videoRef.current?.play();
  };

  // Stop video dan kembalikan ke awal saat pointer keluar
  const handleMouseLeave = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  return (
    <div
      className="group cursor-pointer overflow-hidden rounded-2xl border border-neutral-200 bg-neutral-50 transition hover:border-neutral-400 hover:shadow-md"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Video Container */}
      <div className="relative aspect-[4/3] overflow-hidden bg-neutral-900">
        <video
          ref={videoRef}
          src={item.src}
          loop
          muted
          playsInline
          preload="metadata"
          className="h-full w-full object-cover"
        />
        {/* Label Overlay Nama Kata */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-3 py-2">
          <p className="text-sm font-semibold text-white">{item.label}</p>
        </div>
        {/* Hint Aksi Hover */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100">
          <span className="rounded-full bg-black/40 px-3 py-1 text-xs font-medium text-white backdrop-blur">
            ▶ Hover untuk putar
          </span>
        </div>
      </div>
      {/* Keterangan Hint Gerakan */}
      <div className="px-3 py-2">
        <p className="text-xs text-neutral-500">{item.hint}</p>
      </div>
    </div>
  );
}

export function GestureGuide() {
  return (
    <Card>
      {/* Header Bagian Contoh Panduan */}
      <div className="mb-4 flex items-center gap-2">
        <BookOpen size={18} />
        <h2 className="text-lg font-semibold tracking-tight">
          Gesture Reference
        </h2>
        <span className="ml-auto text-xs text-neutral-400">
          Hover video untuk melihat gerakan
        </span>
      </div>

      {/* Grid List Video Referensi */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-4">
        {GESTURES.map((item) => (
          <GestureCard key={item.label} item={item} />
        ))}
      </div>
    </Card>
  );
}
