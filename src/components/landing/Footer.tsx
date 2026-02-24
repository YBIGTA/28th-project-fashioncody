"use client";

import { Sparkles } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-300">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 p-2">
                <Sparkles className="size-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">OOTD AI</span>
            </div>
            <p className="text-sm text-slate-400">
              AI 기반 스마트 코디 추천 서비스로 매일 새로운 스타일을
              발견하세요.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">제품</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#features" className="hover:text-white transition-colors">
                  기능
                </a>
              </li>
              <li>
                <a href="#gallery" className="hover:text-white transition-colors">
                  갤러리
                </a>
              </li>
              <li>
                <a href="#about" className="hover:text-white transition-colors">
                  소개
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-white font-semibold mb-4">회사</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  팀
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  블로그
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  채용
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-white font-semibold mb-4">지원</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  도움말
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  문의하기
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  개인정보처리방침
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-slate-800 mt-8 pt-8 text-center text-sm text-slate-400">
          <p>© 2024 OOTD AI. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
