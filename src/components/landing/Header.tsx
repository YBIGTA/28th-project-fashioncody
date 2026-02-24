"use client";

import { useState, useEffect } from "react";
import { Menu, X, Sparkles, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function Header() {
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
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 p-2">
              <Sparkles className="size-5 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              OOTD AI
            </span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <a
              href="#features"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors"
            >
              기능
            </a>
            <a
              href="#gallery"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors"
            >
              갤러리
            </a>
            <a
              href="#about"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors"
            >
              소개
            </a>
            <Button variant="ghost" size="sm" className="gap-2">
              <User className="size-4" />
              로그인
            </Button>
            <Button size="sm" className="rounded-xl">
              시작하기
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
            <a
              href="#features"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors py-2"
              onClick={() => setMobileMenuOpen(false)}
            >
              기능
            </a>
            <a
              href="#gallery"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors py-2"
              onClick={() => setMobileMenuOpen(false)}
            >
              갤러리
            </a>
            <a
              href="#about"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors py-2"
              onClick={() => setMobileMenuOpen(false)}
            >
              소개
            </a>
            <div className="flex gap-2 pt-2">
              <Button variant="ghost" size="sm" className="flex-1 gap-2">
                <User className="size-4" />
                로그인
              </Button>
              <Button size="sm" className="flex-1 rounded-xl">
                시작하기
              </Button>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
}
