// KOINONIA Assistant Service Worker v1.0.2
const CACHE_NAME = 'koinonia-cache-v1.0.2';
const STATIC_ASSETS = [
    '/manifest.json',
    '/icons/icon-192.png',
    '/icons/icon-512.png',
    '/icons/icon.svg'
];

// Install: Pre-cache essential icons and manifest
self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS).catch(err => {
                console.warn('[ServiceWorker] Pre-caching warning:', err);
            });
        })
    );
});

// Activate: Clean all old caches immediately and claim clients
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cache) => {
                    if (cache !== CACHE_NAME) {
                        console.log('[ServiceWorker] Purging old cache:', cache);
                        return caches.delete(cache);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch: Network-First for HTML Navigation, Network-Only for API, Cache-First for Static
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // 1. Dynamic API, RPC, Login or Non-GET Requests -> Network Only
    if (url.pathname.startswith('/api/') || 
        url.pathname.startswith('/desk') || 
        url.pathname.startswith('/app') ||
        url.pathname.startswith('/login') ||
        event.request.method !== 'GET') {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(JSON.stringify({
                    error: "Offline: Could not connect to KOINONIA server. Please check your network connection." 
                }), {
                    headers: { 'Content-Type': 'application/json' }
                });
            })
        );
        return;
    }

    // 2. HTML Navigation (Page Loads) -> Network First with Cache Fallback
    if (event.request.mode === 'navigate' || 
        event.request.destination === 'document' ||
        (url.pathname === '/' || url.pathname === '/koinonia_chat')) {
        event.respondWith(
            fetch(event.request)
                .then((networkResponse) => {
                    if (networkResponse && networkResponse.status === 200) {
                        const clone = networkResponse.clone();
                        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                    }
                    return networkResponse;
                })
                .catch(() => caches.match(event.request))
        );
        return;
    }

    // 3. Static Assets (Icons, Styles, Scripts)tested -> Cache-First with Background Revalidation
    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            const fetchPromise = fetch(event.request).then((networkResponse) => {
                if (networkResponse && networkResponse.status === 200) {
                    const responseClone = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                    });
                }
                return networkResponse;
            }).catch(() => cachedResponse);

            return cachedResponse || fetchPromise;
        })
    );
});
