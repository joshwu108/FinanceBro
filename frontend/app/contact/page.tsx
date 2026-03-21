"use client"

import { useState } from "react"
import { Navbar } from "@/components/site/navbar"
import { Footer } from "@/components/site/footer"
import { Send, Github, Mail, MapPin } from "lucide-react"

export default function ContactPage() {
  const [submitted, setSubmitted] = useState(false)

  return (
    <div className="min-h-screen bg-[#0B0F14] text-[#E6EDF3] flex flex-col">
      <Navbar />

      <section className="px-6 py-20 flex-1">
        <div className="max-w-4xl mx-auto">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
            Contact
          </span>
          <h1 className="mt-3 text-3xl md:text-4xl font-bold">
            Get in <span className="text-[#00FF9C]">touch</span>
          </h1>
          <p className="mt-4 text-sm text-[#8B949E] max-w-lg">
            Questions about the platform, interested in contributing, or looking to collaborate
            on quantitative research? Reach out below.
          </p>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-12">
            {/* Contact Form */}
            <div>
              {submitted ? (
                <div className="p-8 bg-[#11161C] border border-[#1E2A38] rounded-sm text-center">
                  <div className="w-10 h-10 rounded-full bg-[#00FF9C]/10 flex items-center justify-center mx-auto mb-4">
                    <Send className="w-4 h-4 text-[#00FF9C]" />
                  </div>
                  <p className="text-sm text-[#E6EDF3] font-semibold">Message sent</p>
                  <p className="text-xs text-[#8B949E] mt-1">We&apos;ll get back to you shortly.</p>
                </div>
              ) : (
                <form
                  onSubmit={(e) => {
                    e.preventDefault()
                    setSubmitted(true)
                  }}
                  className="flex flex-col gap-4"
                >
                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-[#8B949E] mb-1.5">
                      Name
                    </label>
                    <input
                      type="text"
                      required
                      className="w-full px-3 py-2 bg-[#11161C] border border-[#1E2A38] rounded-sm text-xs text-[#E6EDF3] placeholder-[#8B949E]/50 focus:outline-none focus:border-[#4CC9F0] transition-colors"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-[#8B949E] mb-1.5">
                      Email
                    </label>
                    <input
                      type="email"
                      required
                      className="w-full px-3 py-2 bg-[#11161C] border border-[#1E2A38] rounded-sm text-xs text-[#E6EDF3] placeholder-[#8B949E]/50 focus:outline-none focus:border-[#4CC9F0] transition-colors"
                      placeholder="you@example.com"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-[#8B949E] mb-1.5">
                      Subject
                    </label>
                    <select
                      className="w-full px-3 py-2 bg-[#11161C] border border-[#1E2A38] rounded-sm text-xs text-[#E6EDF3] focus:outline-none focus:border-[#4CC9F0] transition-colors"
                    >
                      <option value="general">General Inquiry</option>
                      <option value="internship">Internship Opportunities</option>
                      <option value="research">Research Collaboration</option>
                      <option value="bug">Bug Report</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-[#8B949E] mb-1.5">
                      Message
                    </label>
                    <textarea
                      required
                      rows={5}
                      className="w-full px-3 py-2 bg-[#11161C] border border-[#1E2A38] rounded-sm text-xs text-[#E6EDF3] placeholder-[#8B949E]/50 focus:outline-none focus:border-[#4CC9F0] transition-colors resize-none"
                      placeholder="Your message..."
                    />
                  </div>
                  <button
                    type="submit"
                    className="flex items-center justify-center gap-2 px-6 py-2.5 bg-[#00FF9C] text-[#0B0F14] text-xs font-bold uppercase tracking-wider rounded-sm hover:bg-[#00FF9C]/90 transition-colors w-fit"
                  >
                    <Send className="w-3.5 h-3.5" />
                    Send Message
                  </button>
                </form>
              )}
            </div>

            {/* Contact Info */}
            <div className="flex flex-col gap-6">
              <div className="p-6 bg-[#11161C] border border-[#1E2A38] rounded-sm">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#E6EDF3] mb-4">
                  Reach us directly
                </h3>
                <div className="flex flex-col gap-4">
                  <div className="flex items-center gap-3">
                    <Mail className="w-4 h-4 text-[#4CC9F0] shrink-0" />
                    <span className="text-xs text-[#8B949E]">team@financebro.dev</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Github className="w-4 h-4 text-[#4CC9F0] shrink-0" />
                    <span className="text-xs text-[#8B949E]">github.com/financebro</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <MapPin className="w-4 h-4 text-[#4CC9F0] shrink-0" />
                    <span className="text-xs text-[#8B949E]">San Francisco, CA</span>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-[#11161C] border border-[#1E2A38] rounded-sm">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#E6EDF3] mb-3">
                  Internship inquiries
                </h3>
                <p className="text-xs text-[#8B949E] leading-relaxed">
                  Interested in quantitative research or software engineering internships?
                  Select &ldquo;Internship Opportunities&rdquo; in the subject dropdown and include
                  your background, areas of interest, and any relevant projects.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}
