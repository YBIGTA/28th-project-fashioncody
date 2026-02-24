"use client";

import { useState, useEffect } from "react";
import { Menu, X, Sparkles, ArrowLeft, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import Link from "next/link";

export default function OotdHeader() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        scrolled
          ? "bg-white/95 backdrop-blur-md border-b border-border/40 shadow-md"
          : "bg-white/80 backdrop-blur-md border-b border-border/20 shadow-sm"
      )}
    >
      <nav className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo & Back Button */}
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 group">
              <div className="rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 p-2 group-hover:scale-105 transition-transform">
                <Sparkles className="size-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                OOTD AI
              </span>
            </Link>
            <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
              <span>/</span>
              <span className="font-medium text-foreground">코디 추천</span>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="gap-2 rounded-xl">
                <Home className="size-4" />
                홈으로
              </Button>
            </Link>
            <Button variant="ghost" size="sm" className="gap-2 rounded-xl">
              내 옷장
            </Button>
            <Button variant="ghost" size="sm" className="gap-2 rounded-xl">
              히스토리
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2 rounded-lg hover:bg-accent transition-colors"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="size-5" />
            ) : (
              <Menu className="size-5" />
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        <div
          className={cn(
            "md:hidden overflow-hidden transition-all duration-300",
            mobileMenuOpen ? "max-h-64 pb-4" : "max-h-0"
          )}
        >
          <div className="flex flex-col gap-3 pt-4">
            <Link href="/" onClick={() => setMobileMenuOpen(false)}>
              <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
                <Home className="size-4" />
                홈으로
              </Button>
            </Link>
            <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
              내 옷장
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
              히스토리
            </Button>
          </div>
        </div>
      </nav>
    </header>
  );
}
