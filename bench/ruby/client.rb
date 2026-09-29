# frozen_string_literal: true

# The RubyLLM side of the benchmark: the same cases, in the same order and shape, as
# bench/src/bin/client.rs. Keep the two in step. Every case prints one JSON line.
#
#   ruby client.rb <case> [key=value ...]
#
# Concurrency uses Async fibers (RubyLLM's recommended model, docs/_advanced/async.md) unless
# mode=threads is given.

STARTED = Process.clock_gettime(Process::CLOCK_MONOTONIC)

require 'bundler/setup'
require 'json'
require 'ruby_llm'

CASE = ARGV.shift.to_s
ARGS = ARGV.to_h { |a| a.split('=', 2) }

def arg(key, default) = Integer(ARGS.fetch(key, default))
def base = ARGS.fetch('base', 'http://127.0.0.1:8765')
def now = Process.clock_gettime(Process::CLOCK_MONOTONIC)
def ms(seconds) = seconds * 1000.0

# A context pointing Anthropic at the mock; the path carries the mock's delay and chunk knobs.
def context(delay, chunks)
  RubyLLM.context do |c|
    c.anthropic_api_base = "#{base}/d/#{delay}/n/#{chunks}"
    c.anthropic_api_key = 'bench'
    c.max_retries = 0
  end
end

def chat(ctx) = ctx.chat(model: 'claude-haiku-4-5', provider: :anthropic)

class Weather < RubyLLM::Tool
  description 'Gets current weather for a location'
  parameter :latitude, description: 'Latitude (e.g., 52.5200)'
  parameter :longitude, description: 'Longitude (e.g., 13.4050)'

  def execute(latitude:, longitude:)
    "Current weather at #{latitude}, #{longitude}: 15°C, Wind: 10 km/h"
  end
end

def percentile(sorted, pct)
  return 0.0 if sorted.empty?

  sorted[((pct / 100.0) * (sorted.size - 1)).round.clamp(0, sorted.size - 1)]
end

def summary(samples)
  sorted = samples.sort
  { n: sorted.size, p50_ms: percentile(sorted, 50), p99_ms: percentile(sorted, 99),
    mean_ms: sorted.sum / [sorted.size, 1].max }
end

def rss_kib = File.read('/proc/self/status')[/^VmRSS:\s+(\d+)/, 1].to_i

def concurrently(count, &block)
  if ARGS['mode'] == 'threads'
    Array.new(count) { |i| Thread.new(i, &block) }.map(&:value)
  else
    require 'async'
    Sync { |task| Array.new(count) { |i| task.async { block.call(i) } }.map(&:wait) }
  end
end

result =
  case CASE
  when 'first'
    answer = chat(context(0, 1)).ask("What's 2 + 2?")
    raise "unexpected #{answer.content.inspect}" unless answer.content == '2 + 2 = 4'

    { first_answer_ms: ms(now - STARTED), rss_kib: rss_kib }
  when 'overhead'
    delay = arg('delay', 0)
    ctx = context(delay, 1)
    n = arg('n', 2000)
    50.times { chat(ctx).ask("What's 2 + 2?") }
    samples = Array.new(n) do
      t = now
      answer = chat(ctx).ask("What's 2 + 2?")
      elapsed = ms(now - t) - delay
      raise 'wrong answer' unless answer.content == '2 + 2 = 4'

      elapsed
    end
    summary(samples)
  when 'concurrent'
    delay = arg('delay', 50)
    chats = arg('chats', 100)
    rounds = arg('rounds', 5)
    chat(context(0, 1)).ask("What's 2 + 2?")
    ctx = context(delay, 1)
    t = now
    messages = concurrently(chats) do
      c = chat(ctx)
      rounds.times { c.ask("What's 2 + 2?") }
      render_started = now
      c.render
      [c.messages.size, ms(now - render_started)]
    end
    final_render_ms = messages.last[1]
    messages = messages.sum(&:first)
    raise "lost messages: #{messages}" unless messages == chats * rounds * 2

    elapsed = now - t
    { requests: chats * rounds, rounds: rounds, elapsed_s: elapsed, req_per_s: chats * rounds / elapsed,
      ideal_req_per_s: chats * 1000.0 / [delay, 1].max, rss_kib: rss_kib, final_render_ms: final_render_ms, mode: ARGS.fetch('mode', 'async') }
  when 'stream'
    chunks = arg('chunks', 1000)
    n = arg('n', 50)
    ctx = context(0, chunks)
    3.times { chat(ctx).ask('Count from 1 to 3') { |_| nil } }
    seen = 0
    samples = Array.new(n) do
      t = now
      answer = chat(ctx).ask('Count from 1 to 3') { |chunk| seen += 1 unless chunk.content.to_s.empty? }
      per_chunk = ms(now - t) * 1000.0 / chunks
      raise 'short stream' unless answer.content.size == 5 * chunks

      per_chunk
    end
    raise "saw #{seen} chunks" unless seen == n * chunks

    summary(samples).merge(unit: 'us_per_chunk')
  when 'tools'
    n = arg('n', 200)
    ctx = context(0, 1)
    calls = 0
    run = lambda do
      c = chat(ctx).with_tools(Weather).before_tool_call { |_| calls += 1 }
      answer = c.ask("What's the weather in Berlin? (52.5200, 13.4050)")
      raise 'wrong answer' unless answer.content.start_with?('The current weather in Berlin')
      raise "#{c.messages.size} messages" unless c.messages.size == 8
    end
    20.times { run.call }
    samples = Array.new(n) do
      t = now
      run.call
      ms(now - t)
    end
    raise "#{calls} tool calls" unless calls == (n + 20) * 3

    summary(samples)
  when 'memory'
    chats = arg('chats', 100)
    delay = arg('delay', 2000)
    chat(context(0, 1)).ask("What's 2 + 2?")
    GC.start
    baseline = rss_kib
    ctx = context(delay, 1)
    in_flight = nil
    sampler = Thread.new do
      sleep delay / 2000.0
      in_flight = rss_kib
    end
    concurrently(chats) { chat(ctx).ask("What's 2 + 2?").content.size }
    sampler.join
    { chats: chats, baseline_rss_kib: baseline, in_flight_rss_kib: in_flight, mode: ARGS.fetch('mode', 'async') }
  when 'render'
    n = arg('n', 2000)
    count = arg('messages', 200)
    c = chat(context(0, 1)).with_instructions('You are a helpful assistant.').with_tools(Weather)
    (count / 2).times do |i|
      c.add_message(role: :user, content: "Question #{i}: what's the weather like in city number #{i} today?")
      c.add_message(role: :assistant,
                    content: "Answer #{i}: it is sunny with a light breeze, around #{10 + (i % 20)} degrees Celsius.")
    end
    bytes = JSON.generate(c.render).bytesize
    50.times { JSON.generate(c.render) }
    samples = Array.new(n) do
      t = now
      JSON.generate(c.render)
      ms(now - t)
    end
    summary(samples).merge(payload_bytes: bytes)
  else
    abort "unknown case #{CASE.inspect}"
  end

puts JSON.generate(impl: 'ruby', case: CASE, yjit: defined?(RubyVM::YJIT) && RubyVM::YJIT.enabled?, result: result)
